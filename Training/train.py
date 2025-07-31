 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.4
Date: 04-05-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import time
import torch


import CardioCascadeNet



class TrainNetwork(CardioCascadeNet.MetaParameters):
    def __init__(self, model, optimizer, loss_function, scheduler_gen, train_loader, valid_loader):         
        super(CardioCascadeNet.MetaParameters, self).__init__()
        self.ds = CardioCascadeNet.DiceLoss()
        self.fdwr = CardioCascadeNet.FileDirectoryWorker()
        self.device = CardioCascadeNet.device

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.scheduler_gen = scheduler_gen
        self.print_model_key

    @property  
    def choose_model_key(self):
        if self.UNET5 is True:
            return self.UNET5_FOLD
        elif self.UNET4 is True and self.UNET5 is False:
            return self.UNET4_FOLD    
        elif self.UNET3 is True and self.UNET4 is False:
            return self.UNET3_FOLD
        elif self.UNET2 is True and self.UNET3 is False:
            return self.UNET2_FOLD
        elif self.UNET1 is True and self.UNET2 is False:
            return self.UNET1_FOLD

    @property
    def model_key(self):
        return self.choose_model_key

    @property
    def print_model_key(self):
        print(f'Model KEY {self.model_key} Was Chosen')

    def get_metrics(self, loader_):
        self.model.eval()

        loss = 0
        num_batches = len(loader_)
        num_layers = len(self.DICT_CLASS)   
        dictionary = {}

        for key in range(1, num_layers): 
            dictionary[f'Dice_{self.DICT_CLASS[key]}'] = 0

        with torch.no_grad():
            
            for inputs, labels, sub_names in loader_:
                inputs, labels, sub_names = inputs.to(self.device), labels.to(self.device), list(sub_names)   

                predict = self.model(inputs)
                loss += self.loss_function(predict, labels)

                predict = torch.argmax(predict, dim = 1)
                labels = torch.argmax(labels, dim = 1)
                
                for key in range(1, num_layers):
                    predict_ = (predict == key)
                    labels_ = (labels == key)

                    dictionary[f'Dice_{self.DICT_CLASS[key]}'] += float(self.ds(predict_, labels_))

        for key in range(1, num_layers):
            dictionary[f'Dice_{self.DICT_CLASS[key]}'] /= num_batches
        
        mean_loss = float((loss / num_batches))

        return mean_loss, dictionary

    def train(self):
        trigger_times, the_last_loss = 0, 100
        best_results = ''
        
        for epoch in range(self.EPOCHS + 1):
            current_results = ''

            time_start_epoch = time.time()
            
            self.model.train()
            
            for inputs, labels, sub_names in self.train_loader:
                inputs, labels, sub_names = inputs.to(self.device), labels.to(self.device), list(sub_names)   
          
                predict = self.model(inputs)
                train_loss = self.loss_function(predict, labels)

                predict = torch.argmax(predict, dim = 1)
                labels = torch.argmax(labels, dim = 1)
                
                # train_loss_02 = 0
                
                # for key in range(1, self.NUM_CLASS):
                #     predict_ = (predict == key)
                #     labels_ = (labels == key)
                #     train_loss_02 += (1 - float(self.ds(predict_, labels_)) * self.CE_WEIGHTS[key])

                # train_loss = train_loss_01 + train_loss_02
                
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            # with warmup_scheduler.dampening():
                # self.scheduler_gen.step()
            self.scheduler_gen.step() #g_mean_train_loss,  g_mean_valid_loss
            
            training = self.get_metrics(self.train_loader)
            validating = self.get_metrics(self.valid_loader)

            # val_loss = validating[0]
            # self.scheduler_gen.step(val_loss)
            
            current_results += f'EPOCH: {epoch}\n'
            current_results += f'TRAIN: Loss = {round(training[0], 3)}'
            
            for key in range(1, self.NUM_CLASS):
                current_results += f' Dice_{self.DICT_CLASS[key]} = ' + str(round(training[1].get(f'Dice_{self.DICT_CLASS[key]}'), 3))
                
            current_results += f'\nVALID: Loss = {round(validating[0], 3)}'
            
            for key in range(1, self.NUM_CLASS):
                current_results += f' Dice_{self.DICT_CLASS[key]} = ' + str(round(validating[1].get(f'Dice_{self.DICT_CLASS[key]}'), 3))
            
            self.fdwr.log_stats(project_name = self.PROJ_NAME, results = current_results)

            if validating[0] > the_last_loss:
                trigger_times += 1
                print('trigger times:', trigger_times)

                if trigger_times >= self.EARLY_STOPPING:
                    print('Early stopping!\nStart to test process.')
                    
                    self.save_hyperparams(best_results)
                    return self.model

            else:
                trigger_times = 0

            if validating[0] <= the_last_loss:
                the_last_loss = validating[0]
                best_results = current_results

                try:
                    checkpoint = torch.load(f'{self.PROJ_NAME}/{self.DATASET_NAME}_model.pth')
                    checkpoint[f'Net_{self.DATASET_NAME}_{self.model_key}'] = {'Model': self.model, 'weights': self.model.state_dict()}
                except:
                    checkpoint = {f'Net_{self.DATASET_NAME}_{self.model_key}': {'Model': self.model, 'weights': self.model.state_dict()}}
 
                torch.save(
                    checkpoint,
                    f'{self.PROJ_NAME}/{self.DATASET_NAME}_model.pth')

                print(f'{self.DATASET_NAME}_model.pth - epoch {epoch} saved!')

            print(current_results)

            time_end_epoch = time.time()
            print(f'Epoch time: {round(time_end_epoch - time_start_epoch)} seconds') 
        
        self.save_hyperparams(best_results)   

        return self.model

    def save_hyperparams(self, best_results: str) -> None:
        hyperparams = f'Trainig date: {time.ctime()}, \n' \
        f'AUGMENTATION: {self.AUGMENTATION}, FREEZE_BN: {self.FREEZE_BN}, PRETRAIN: {self.PRETRAIN}, \n' \
        f'NOISE: {self.NOISE}, EMPTY: {self.EMPTY}, MULTYGAP: {self.MULTYGAP}, \n' \
        f'UNET1: {self.UNET1}, UNET2: {self.UNET2}, UNET3: {self.UNET3}, UNET4: {self.UNET4}, UNET5: {self.UNET5}, \n' \
        f'BGCROPP: {self.BGCROPP}, LVCROPP: {self.LVCROPP}, BGLVCROPP: {self.BGLVCROPP}, SHUFFLE: {self.SHUFFLE}, \n' \
        f'KERNEL: {self.KERNEL}, CROPP_KERNEL: {self.CROPP_KERNEL},' \
        f'CHANNELS: {self.CHANNELS}, LR: {self.LR}, BT_SZ: {self.BT_SZ}, EPOCHS: {self.EPOCHS}, \n' \
        f'DROPOUT: {self.DROPOUT}, FEATURES: {self.FEATURES}, WDC: {self.WDC}, ' \
        f'EARLY_STOPPING: {self.EARLY_STOPPING}, TMAX: {self.TMAX}, CLIP_RATE: {self.CLIP_RATE}, \n' \
        f'SCAR_DICT_CLASS: {self.SCAR_DICT_CLASS}, \nMYOLEVEL_DICT_CLASS: {self.MYOLEVEL_DICT_CLASS}, \n' \
        f'BULLEYE_DICT_CLASS: {self.BULLEYE_DICT_CLASS}, \nTARGET_CE_WEIGHTS: {self.TARGET_CE_WEIGHTS}, \n' \
        f'SCAR_CE_WEIGHTS: {self.SCAR_CE_WEIGHTS}, \nMYOLEVEL_CE_WEIGHTS: {self.MYOLEVEL_CE_WEIGHTS}, \n' \
        f'BULLEYE_CE_WEIGHTS: {self.BULLEYE_CE_WEIGHTS} \n\nBEST_RESULTS: {best_results}\n\n' \

        self.fdwr.log_stats(project_name = f'{self.PROJ_NAME}_hyperparams', results = hyperparams)


