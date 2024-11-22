

from utils.models import *
from utils.tools import *
import os
import imageio
import math
import random
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv
from torch.autograd import Variable

from tqdm.notebook import tqdm
from config import *
from utils.TDA_Image_Process2 import topological_process_img
import glob
from PIL import Image

class Agent():
    def __init__(self, classe, alpha=0.2, nu=3.0, threshold=0.5, num_episodes=15, load=False, model_name='vgg16', n_actions=9):
        # basic settings
        self.n_actions = n_actions                       # total number of actions
        screen_height, screen_width = 224, 224   # size of resized images
        self.GAMMA = 0.900                       # decay weight
        self.EPS = 1                             # initial epsilon value, decayed every epoch
        self.alpha = alpha                       # â‚¬[0, 1]  Scaling factor
        self.nu = nu                             # Reward of Trigger
        self.threshold = threshold               # threshold of IoU to consider as True detection
        self.actions_history = None              # action history vector as record, later initialized in train/predict
        self.steps_done = 0                      # to count how many steps it used to compute the final bdbox
        
        # networks
        self.classe = classe                     # which class this agent is working on
        self.save_path = SAVE_MODEL_PATH         # path to save network
        self.model_name = model_name             # which model to use for feature extractor 'vgg16' or 'resnet50' or ...
        self.feature_extractor = FeatureExtractor(network=self.model_name)
        self.feature_extractor.eval()            # a pre-trained CNN model as feature extractor
        
        if not load:
            self.policy_net = DQN(screen_height, screen_width, self.n_actions, history_length=9)
        else:
            self.policy_net = self.load_network() # policy net - DQN, inputs state vector, outputs q value for each action
        
        self.target_net = DQN(screen_height, screen_width, self.n_actions, history_length=9)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()                    # target net - same DQN as policy net, works as frozen net to compute loss
                                                  # initialize as the same as policy net, use eval to disable Dropout
            
        # training settings
        self.BATCH_SIZE = 128                    # batch size
        self.num_episodes = num_episodes         # number of total episodes
        self.memory = ReplayMemory(10000)        # experience memory object
        self.TARGET_UPDATE = 1                   # frequence of update target net
        self.actions_history = torch.zeros((9, self.n_actions))
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-6)  # optimizer
        
        self.window_size = None
        self.border_width = None
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if use_cuda:
            self.feature_extractor = self.feature_extractor.cuda()
            self.target_net = self.target_net.cuda()
            self.policy_net = self.policy_net.cuda()

    def save_network(self):
        torch.save(self.policy_net, self.save_path + "_" + self.model_name + "_" +self.classe)
        print('Saved')

    def load_network(self):
        if not use_cuda:
            return torch.load(self.save_path + "_" + self.model_name + "_" + self.classe, map_location=torch.device('cpu'))
        return torch.load(self.save_path + "_" + self.model_name + "_" + self.classe)
    
    #############################
    # 1. Functions to compute reward
    def intersection_over_union(self, box1, box2):
        """
        Compute IoU value over two bounding boxes
        Each box is represented by four elements vector: (left, right, top, bottom)
        Origin point of image system is on the top left
        """
        box1_left, box1_right, box1_top, box1_bottom = box1
        box2_left, box2_right, box2_top, box2_bottom = box2
        
        inter_top = max(box1_top, box2_top)
        inter_left = max(box1_left, box2_left)
        inter_bottom = min(box1_bottom, box2_bottom)
        inter_right = min(box1_right, box2_right)
        inter_area = max(((inter_right - inter_left) * (inter_bottom - inter_top)), 0)
        
        box1_area = (box1_right - box1_left) * (box1_bottom - box1_top)
        box2_area = (box2_right - box2_left) * (box2_bottom - box2_top)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

    def compute_reward(self, actual_state, previous_state, ground_truth):
        """
        Compute the reward based on IoU before and after an action (not trigger)
        The reward will be +1 if IoU increases, and -1 if decreases or stops
        ----------
        Argument:
        actual_state   - new bounding box after action
        previous_state - old boudning box
        ground_truth   - ground truth bounding box of current object
        *all bounding boxes comes in four elements vector (left, right, top, bottom)
        ----------
        Return:
        reward         - +1/-1 depends on difference between IoUs
        """
        res = self.intersection_over_union(actual_state, ground_truth) - self.intersection_over_union(previous_state, ground_truth)
        if res <= 0:
            #print('-1')
            return -1
        #print('1')
        return 1
    
    def compute_trigger_reward(self, actual_state, ground_truth):
        """
        Compute the reward based on final IoU before *trigger*
        The reward will be +nu if final IoU is larger than threshold, and -nu if not
        ----------
        Argument:
        actual_state - final bounding box before trigger
        ground_truth - ground truth bounding box of current object
        *all bounding boxes comes in four elements vector (left, right, top, bottom)
        ----------
        Return:
        reward       - +nu/-nu depends on final IoU
        """
        res = self.intersection_over_union(actual_state, ground_truth)
        if res>=self.threshold:
            return self.nu
        return -1*self.nu
      
        
    
    ###########################
    # 2. Functions to get actions 
    def calculate_position_box(self, current_coord, action):
        """
        Calculate new coordinate based on current coordinate and taken action.
        ----------
        Argument:
        current_coord - the current coordinate of this agent, comes in four elements vector (xmin, ymin, xmax, ymax)
        action        - the index of taken action, should be between 0-8
        ----------
        Return:
        new_coord     - the coordinate after taking the action, also four elements vector (xmin, ymin, xmax, ymax)
        """
        xmin, ymin, xmax, ymax = current_coord
    
        width = xmax - xmin
        height = ymax - ymin
        alpha_w = self.alpha * width
        alpha_h = self.alpha * height
    
        if action == 1:  # Right
            xmin += alpha_w
            xmax += alpha_w
        elif action == 2:  # Left
            xmin -= alpha_w
            xmax -= alpha_w
        elif action == 3:  # Up 
            ymin -= alpha_h
            ymax -= alpha_h
        elif action == 4:  # Down
            ymin += alpha_h
            ymax += alpha_h
        elif action == 5:  # Bigger
            xmin -= alpha_w
            ymin -= alpha_h
            xmax += alpha_w
            ymax += alpha_h
        elif action == 6:  # Smaller
            xmin += alpha_w
            ymin += alpha_h
            xmax -= alpha_w
            ymax -= alpha_h
        elif action == 7:  # Fatter
            xmin -= alpha_w
            xmax += alpha_w
        elif action == 8:  # Taller
            ymin -= alpha_h
            ymax += alpha_h
    
        xmin = self.rewrap(xmin)
        ymin = self.rewrap(ymin)
        xmax = self.rewrap(xmax)
        ymax = self.rewrap(ymax)
        
        return [xmin, ymin, xmax, ymax]

    
    def get_best_next_action(self, current_coord, ground_truth):
        """
        Given actions, traverse every possible action, cluster them into positive actions and negative actions
        Then randomly choose one positive actions if exist, or choose one negtive actions anyways
        It is used for epsilon-greedy policy
        ----------
        Argument:
        current_coord - the current coordinate of this agent, should comes in four elements vector (left, right, top, bottom)
        ----------
        Return:
        An action index that represents the best action next
        """
        positive_actions = []
        negative_actions = []
        for i in range(0, self.n_actions):
            new_equivalent_coord = self.calculate_position_box(current_coord, i)
            if i!=0:
                reward = self.compute_reward(new_equivalent_coord, current_coord, ground_truth)
            else:
                reward = self.compute_trigger_reward(new_equivalent_coord, ground_truth)
            
            if reward>1:
                return 0
            if reward>=0:
                positive_actions.append(i)
            else:
                negative_actions.append(i)
        if len(positive_actions)==0:
            return random.choice(negative_actions)
        return random.choice(positive_actions)

    def select_action(self, state, current_coord, ground_truth):
        """
        Select an action during the interaction with environment, using epsilon greedy policy
        This implementation should be used when training
        ----------
        Argument:
        state         - the state varible of current agent, consisting of (o,h), should conform to input shape of policy net
        current_coord - the current coordinate of this agent, should comes in four elements vector (left, right, top, bottom)
        ground_truth  - the groundtruth of current object
        ----------
        Return:
        An action index after conducting epsilon-greedy policy to current state
        """
        sample = random.random()
        # epsilon value is assigned by self.EPS
        eps_threshold = self.EPS
        # self.steps_done is to count how many steps the agent used to get final bounding box
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                if use_cuda:
                    inpu = Variable(state).cuda()
                else:
                    inpu = Variable(state)
                qval = self.policy_net(inpu)
                _, predicted = torch.max(qval.data,1)
                action = predicted[0] # + 1
                try:
                    return action.cpu().numpy()[0]
                except:
                    return action.cpu().numpy()
        else:
            return self.get_best_next_action(current_coord, ground_truth)

    def select_action_model(self, state):
        """
        Select an action during the interaction with environment, using greedy policy
        This implementation should be used when testing
        ----------
        Argument:
        state - the state varible of current agent, consisting of (o,h), should conform to input shape of policy net
        ----------
        Return:
        An action index which is generated by policy net
        """
        with torch.no_grad():
            if use_cuda:
                inpu = Variable(state).cuda()
            else:
                inpu = Variable(state)
            qval = self.policy_net(inpu)
            _, predicted = torch.max(qval.data,1)
            #print("Predicted : "+str(qval.data))
            action = predicted[0] # + 1
            #print(action)
            return action
            
    def rewrap(self, coord):
        """
        A small function used to ensure every coordinate is inside [0,224]
        """
        return min(max(coord,0), 224)
    
    
    
    ########################
    # 3. Functions to form input tensor to policy network
    def get_features(self, image, dtype=FloatTensor):
        """
        Use feature extractor (a pre-trained CNN model) to transform an image to feature vectors
        """
        # Ensure the image has the correct shape (B, C, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension if it's missing
        elif image.dim() == 5:
            image = image.squeeze(0)  # Remove extra dimension if it's present
        
        # Change it to torch Variable
        image = Variable(image).type(dtype)
        if use_cuda:
            image = image.cuda()
        feature = self.feature_extractor(image)
        return feature.data
    
    def update_history(self, action):
        """
        Update action history vector with a new action
        ---------
        Argument:
        action         - a new taken action that should be updated into action history
        ---------
        Return:
        actions_history - a tensor of (9x9), encoding action history information
        """
        action_vector = torch.zeros(self.n_actions)
        action_vector[action] = 1
        for i in range(0,8,1):
            self.actions_history[i][:] = self.actions_history[i+1][:]
        self.actions_history[8][:] = action_vector[:]
        return self.actions_history
    
    def compose_state(self, image, dtype=FloatTensor):
        """
        Compose image feature and action history to a state variable
        ---------
        Argument:
        image - raw image data
        ---------
        state - a state variable, which is concatenation of image feature vector and action history vector
        """
        image_feature = self.get_features(image, dtype)
        image_feature = image_feature.view(1,-1)
        history_flatten = self.actions_history.view(1,-1).type(dtype)
        state = torch.cat((image_feature, history_flatten), 1)
        return state
    
    
    
    ########################
    # 4. Main training functions
    def optimize_model(self, verbose):
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        
        # Check if there are any non-final states
        if non_final_mask.any():
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        else:
            # If all states are terminal, we don't need to compute next state values
            non_final_next_states = None
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward).unsqueeze(1)
    
        if use_cuda:
            non_final_mask = non_final_mask.cuda()
            if non_final_next_states is not None:
                non_final_next_states = non_final_next_states.cuda()
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
    
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.BATCH_SIZE, 1)
        if use_cuda:
            next_state_values = next_state_values.cuda()
        
        if non_final_next_states is not None:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1)
        
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        
        if verbose:
            print("Loss:{}".format(loss))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train(self, train_loader, verbose=False):
        for i_episode in range(self.num_episodes):
            print(f"Episode {i_episode}")
            for image, target in train_loader:
                original_image = image.clone()
                ground_truth_boxes = [[x.item() for x in target]] 
                ground_truth = ground_truth_boxes[0]
                
                # TDA initialization
                np_image = image.squeeze(0).numpy().transpose(1, 2, 0)
                np_image = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)
                
                delta = math.sqrt(np_image.shape[0]**2 + np_image.shape[1]**2)
                self.window_size = int(delta / 25)
                self.border_width = int(delta / 100)
                
                t, b, l, r = topological_process_img(None, np_image, window_size=self.window_size, border_width=self.border_width)
                self.current_coord = [l, t, r, b]  # [xmin, ymin, xmax, ymax]
                
                self.actions_history = torch.zeros((9, self.n_actions))
                new_image = image
                state = self.compose_state(image)
                
                new_equivalent_coord = self.current_coord
                
                done = False
                t = 0  # Step counter
                
                while not done:
                    t += 1
                    action = self.select_action(state, self.current_coord, ground_truth)
                    
                    if action == 0:
                        next_state = None
                        closest_gt = self.get_max_bdbox(ground_truth_boxes, self.current_coord)
                        reward = self.compute_trigger_reward(self.current_coord, closest_gt)
                        done = True
                    else:
                        self.actions_history = self.update_history(action)
                        new_equivalent_coord = self.calculate_position_box(self.current_coord, action)
                        new_xmin = self.rewrap(int(new_equivalent_coord[0]) - 16)
                        new_ymin = self.rewrap(int(new_equivalent_coord[1]) - 16)
                        new_xmax = self.rewrap(int(new_equivalent_coord[2]) + 16)
                        new_ymax = self.rewrap(int(new_equivalent_coord[3]) + 16)
                        
                        new_image = original_image[:, new_ymin:new_ymax, new_xmin:new_xmax]
                        try:
                            new_image = self.transform(new_image)
                        except ValueError:
                            break
                        
                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox(ground_truth_boxes, new_equivalent_coord)
                        reward = self.compute_reward(new_equivalent_coord, self.current_coord, closest_gt)
                        self.current_coord = new_equivalent_coord

                    if t == 20:
                        done = True
                    
                    self.memory.push(state, int(action), next_state, reward)
                    state = next_state
                    image = new_image
                    
                    self.optimize_model(verbose)
            
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if i_episode < 5:
                self.EPS -= 0.18
            
            self.save_network()
            print('Episode complete.')
    
    
    def get_max_bdbox(self, ground_truth_boxes, actual_coordinates):
        """
        A simple function to hanlde more than 1 object in a picture
        It will compute IoU over every ground truth box and current coordinate and choose the largest one
        And return the corresponding ground truth box as actual ground truth
        """
        max_iou = False
        max_gt = []
        for gt in ground_truth_boxes:
            iou = self.intersection_over_union(actual_coordinates, gt)
            if max_iou == False or max_iou < iou:
                max_iou = iou
                max_gt = gt
        return max_gt
    
    
    
    
    ########################
    # 5. Predict and evaluate functions
    def predict_image(self, image, plot=False, verbose=False):
        self.policy_net.eval()
        
        np_image = image.squeeze(0).numpy().transpose(1, 2, 0)
        np_image = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)
        
        delta = math.sqrt(np_image.shape[0]**2 + np_image.shape[1]**2)
        self.window_size = int(delta/25)
        self.border_width = int(delta/100)
        
        t, b, l, r = topological_process_img(None, np_image, window_size=self.window_size, border_width=self.border_width)
        self.current_coord = [l, t, r, b]  # [xmin, ymin, xmax, ymax]
        
        original_image = image.clone()
        self.actions_history = torch.zeros((9,self.n_actions))
        state = self.compose_state(image)
        
        new_image = image
        steps = 0
        done = False
        cross_flag = True
        
        while not done:
            steps += 1
            action = self.select_action_model(state)
            
            if action == 0:
                next_state = None
                new_equivalent_coord = self.current_coord
                done = True
            else:
                self.actions_history = self.update_history(action)
                new_equivalent_coord = self.calculate_position_box(self.current_coord, action)
                
                new_xmin = self.rewrap(int(new_equivalent_coord[0])-16)
                new_ymin = self.rewrap(int(new_equivalent_coord[1])-16)
                new_xmax = self.rewrap(int(new_equivalent_coord[2])+16)
                new_ymax = self.rewrap(int(new_equivalent_coord[3])+16)
                
                new_image = original_image[:, new_ymin:new_ymax, new_xmin:new_xmax]
                try:
                    new_image = self.transform(new_image)
                except ValueError:
                    break            
                
                next_state = self.compose_state(new_image)
                self.current_coord = new_equivalent_coord
            
            if steps == 40:
                done = True
                cross_flag = False
            
            state = next_state
            image = new_image
            
            if verbose:
                print(f"Iteration:{steps} - Action:{action} - Position:{new_equivalent_coord}")
            
            if plot:
                show_new_bdbox(original_image, new_equivalent_coord, color='b', count=steps)
        
        return new_equivalent_coord
    
    def predict_multiple_objects(self, image, plot=False, verbose=False):
        """
        Iteratively predict multiple objects, when one object is detected, draw a cross on it
        Perform up to 100 steps
        """
        
        new_image = image.clone()
        all_steps = 0
        bdboxes = []   
        
        while 1:
            bdbox, cross_flag, steps = self.predict_image(new_image, plot, verbose)
            bdboxes.append(bdbox)
            
            if cross_flag:
                mask = torch.ones((224,224))
                middle_x = round((bdbox[0] + bdbox[1])/2)
                middle_y = round((bdbox[2] + bdbox[3])/2)
                length_x = round((bdbox[1] - bdbox[0])/8)
                length_y = round((bdbox[3] - bdbox[2])/8)

                mask[middle_y-length_y:middle_y+length_y,int(bdbox[0]):int(bdbox[1])] = 0
                mask[int(bdbox[2]):int(bdbox[3]),middle_x-length_x:middle_x+length_x] = 0

                new_image *= mask
                
            all_steps += steps
                
            if all_steps > 100:
                break
                    
        return bdboxes
        
    
    def evaluate(self, dataset):
        ground_truth_boxes = []
        predicted_boxes = []
        print("Predicting boxes...")
        for idx in range(len(dataset)):
            image, target = dataset[idx]
            annot = target['annotation']['object'][0]['bndbox']
            gt_box = [
                annot['xmin'].item() * 224,
                annot['ymin'].item() * 224,
                annot['xmax'].item() * 224,
                annot['ymax'].item() * 224
            ]
            bbox = self.predict_image(image)
            ground_truth_boxes.append([gt_box])
            predicted_boxes.append(bbox)
        print("Computing recall and ap...")
        stats = eval_stats_at_threshold(predicted_boxes, ground_truth_boxes, thresholds=[0.2,0.3,0.4,0.5])
        print("Final result : \n"+str(stats))
        return stats