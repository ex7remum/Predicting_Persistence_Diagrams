import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch

import models.pd_decoders
import models.generators
import models.pi_models
import models.image_encoders
import models.data_to_pd_models
import train_models

from torch.nn import MSELoss, CrossEntropyLoss
from persim import PersistenceImager
from gudhi.representations.vector_methods import PersistenceImage as PersistenceImageGudhi
import wandb
import random

from get_datasets import get_loaders_by_name
import utils.metrics
from utils.loss_functions import ChamferLoss, HungarianLoss, SlicedWasserstein

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    set_random_seed(54)
    batch_size = 128
    n_runs = 3
    final_metrics = {}
    for n_repeat in range(n_runs):
        #run = wandb.init(
        #        # Set the project where this run will be logged
        #        project="Data to pd Orbit5k",
        #        # Track hyperparameters and run metadata
        #)

        ## datasets
        #datasets = ['Obayashi-Hiraoka', 'UIUC', 'KTH', 'Outex']
        # datasets = ['Obayashi-Hiraoka', 'UIUC']
        datasets = ["Obayashi-Hiraoka"]

        for dataset_name in datasets:
            in_channels = 1
            # load dataset
            dataloader_train, dataloader_test, num_classes, n_max, sigma, im_range = get_loaders_by_name(dataset_name,
                                                                                                         batch_size,
                                                                                                    data_type='image')
                
            # calculate classification accuracy on original PI's
            name = 'original_pi' + '_' + dataset_name
            acc_logreg, acc_rfc = utils.metrics.logreg_and_rfc_acc(dataloader_train, dataloader_test, name, 'default')
            
            if n_repeat == 0:
                final_metrics[name + '_rfc_acc'] = []
                final_metrics[name + '_logreg_acc'] = []
            
            final_metrics[name + '_rfc_acc'].append(acc_rfc)
            final_metrics[name + '_logreg_acc'].append(acc_logreg)
            
            # define our model
            name = 'size_pred_simplecnn_' + dataset_name
            size_predictor = models.image_encoders.SimpleCNNEncoder(in_channels=1, n_out_enc=1).to(device)
            
            size_predictor = train_models.train_size_predictor(size_predictor, dataloader_train, 
                                                                   dataloader_test, name)

            # add other generators and encoders
            latent_dim = 1024
            simple_cnn = models.image_encoders.SimpleCNNEncoder(in_channels=in_channels, 
                                                                n_out_enc=latent_dim).to(device)
            
            top_n = models.generators.TopNGenerator(set_channels=2,
                                                    cosine_channels=32, 
                                                    max_n=n_max + 20, 
                                                    latent_dim=latent_dim).to(device)
            
            mlp_gen = models.generators.MLPGenerator(set_channels=2,
                                                    max_n=n_max+20,
                                                    mlp_gen_hidden=512,
                                                    n_layers=2,
                                                    latent_dim=latent_dim).to(device)
            
            encoders = [(simple_cnn, 'SimpleCNN')]
            generators = [(top_n, 'TopN')]
            
            for encoder, encoder_name in encoders:
                for generator, generator_name in generators:
            
                    decoder = models.pd_decoders.TransformerDecoder(n_in=2,
                                                                    latent_dim=latent_dim,
                                                                    fc_dim=1024, 
                                                                    num_heads=8,
                                                                    num_layers=5, 
                                                                    n_out=2,
                                                                    generator=generator,
                                                                    n_out_lin=512, 
                                                                    n_hidden=1024, 
                                                                    num_layers_lin=2,
                                                                    dropout = 0.1,
                                                                    use_conv=False).to(device)

                    model = models.data_to_pd_models.OneShotPd(encoder, 
                                                               decoder, 
                                                               size_predictor=size_predictor,
                                                               n_max=n_max).to(device)

                    # train our model
                    name = encoder_name + '_' + generator_name + '_' + dataset_name
                    
                    
                    model = train_models.train_full_model(model, dataloader_train, dataloader_test, name, n_repeat,
                                                             n_epochs=200)
                    torch.save(model.state_dict(), 
                                   './pretrained_models/final_full_model_' + name + '_run_' + str(n_repeat) + ".pt")
                    
                    W2_res = utils.metrics.calc_gudhi_W2_dist(model, dataloader_test)
                    btlnck_res = utils.metrics.calc_gudhi_bottleneck_dist(model, dataloader_test)
                    
                    if n_repeat == 0:
                        final_metrics[name + '_W2'] = []
                        final_metrics[name + '_bottleneck_distance'] = []
                        final_metrics[name + '_runtime'] = []
                        final_metrics[name + '_rfc_acc'] = []
                        final_metrics[name + '_logreg_acc'] = []
                        final_metrics[name + '_pred_pd_acc'] = []
                        final_metrics[name + '_real_pd_acc'] = []
                        final_metrics[name + '_PIE'] = []
                    
                    model_time = utils.metrics.calc_inference_time(model, dataloader_test)
                    print("Full model runtime " + name, model_time)
                    #wandb.log({"Full model runtime " + name: model_time})
                    
                    final_metrics[name + '_W2'].append(W2_res)
                    final_metrics[name + '_bottleneck_distance'].append(btlnck_res)
                    final_metrics[name + '_runtime'].append(model_time)

                    # train classificators
                    model_class_on_real = train_models.train_classifier_on_pds(dataloader_train, 
                                                                               dataloader_test,
                                                                               n_classes=num_classes, 
                                                                               real_pd=True, 
                                                                               model=None, 
                                                                               name=name,
                                                                               n_epochs=200)
                    
                    model_class_on_pred = train_models.train_classifier_on_pds(dataloader_train, 
                                                                               dataloader_test,
                                                                               n_classes=num_classes, 
                                                                               real_pd=False, 
                                                                               model=model, 
                                                                               name=name,
                                                                               n_epochs=200)
                    
                    acc_real = utils.metrics.calculate_accuracy(model, model_class_on_real, dataloader_test, 
                                                                on_real=True)
                    
                    acc_pred = utils.metrics.calculate_accuracy(model, model_class_on_pred, dataloader_test, 
                                                                on_real=False)
                    
                    
                    final_metrics[name + '_pred_pd_acc'].append(acc_pred)
                    final_metrics[name + '_real_pd_acc'].append(acc_real)
                    # calculate classification accuracy on PI's
                    # directly computed from our predicted PD's
                    # change pimgr to be in one style
                    pimgr = PersistenceImageGudhi(bandwidth=sigma, 
                                                  resolution=[50, 50], 
                                                  weight=lambda x: (x[1])**2, 
                                                  im_range=im_range)
                    
                    PIE = utils.metrics.calc_pie(model, dataloader_test, pimgr)
                    
                    final_metrics[name + '_PIE'].append(PIE)
                    print("PIE " + name, PIE, pimgr)
                    #wandb.log({"PIE " + name: model_time})
                    
                    acc_logreg, acc_rfc = utils.metrics.logreg_and_rfc_acc(dataloader_train, dataloader_test, name,
                                                                          'from_pd', model, pimgr)
                    
                    final_metrics[name + '_logreg_acc'].append(acc_logreg)
                    final_metrics[name + '_rfc_acc'].append(acc_rfc)

            # train PI predictor models
            pi_net = models.pi_models.PI_Net(in_channels=in_channels).to(device)
            
            image_pi_models = [(pi_net, 'PINet')]
            
            for pi_model, pi_model_name in image_pi_models:

                name = pi_model_name + '_' + dataset_name
                
                if n_repeat == 0:
                    final_metrics[name + '_runtime'] = []
                    final_metrics[name + '_rfc_acc'] = []
                    final_metrics[name + '_logreg_acc'] = []
                    final_metrics[name + '_PIE'] = []
                
                pi_model = train_models.train_pi_model(pi_model, dataloader_train, dataloader_test,
                                                       crit=MSELoss(), name=name, n_epochs=200)
                torch.save(pi_model.state_dict(), 
                           './pretrained_models/' + name + '_run_' + str(n_repeat) + ".pt")

                model_time = utils.metrics.calc_inference_time(pi_model, dataloader_test)
                final_metrics[name + '_runtime'].append(model_time)
                print(pi_model_name + " runtime " + dataset_name, model_time)
                #wandb.log({pi_model_name + " runtime " + dataset_name : model_time})
                
                PIE = utils.metrics.calc_pie(pi_model, dataloader_test)
                final_metrics[name + '_PIE'].append(PIE)
                print("PIE " + name, PIE)
                #wandb.log({"PIE " + name: model_time})

                # compare acc of PIs by different models on logreg and random forest
                acc_logreg, acc_rfc = utils.metrics.logreg_and_rfc_acc(dataloader_train, dataloader_test, 
                                                                       name, 'pi_model', pi_model)
                #wandb.log({"log reg acc " + name: acc_logreg, "rand forest acc " +  name: acc_rfc})
                print('Classificaion accuracy: log reg: {0}, rfc: {1}'.format(acc_logreg, acc_rfc), name)
                final_metrics[name + '_logreg_acc'].append(acc_logreg)
                final_metrics[name + '_rfc_acc'].append(acc_rfc)
            path_cur = './results/temp_images_run_{}.pt'.format(n_repeat)
            torch.save(final_metrics, path_cur)
            