import torch
from torchvision import datasets, transforms
import numpy as np
from itertools import combinations
from model import SplitNN, models_generate
from random import choice
from data_dis import Distribute_data
import os
import datetime
from tqdm import tqdm
import argparse
import json
import random
import logging
from util import *

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def setup_data(config, data_owners):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=True)
    distributed_testloader = Distribute_data(data_owners=data_owners, data_loader=testloader)
    return distributed_testloader

def setup_models(data_owners):
    input_size = 28 * 4
    hidden_size = 64 * 7
    models = models_generate(data_owners, input_size, hidden_size)
    return models

def load_models(splitnn, load_dir):
    for location, model in splitnn.models.items():
        model_path = os.path.join(load_dir, f'{location}_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
    print(f"Models loaded from {load_dir}")


def main(args):
    setup_logging()
    config = load_config(args.config)
    
    seed = random.randint(10, 1000)
    logging.info(f"Using seed: {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    clients_num = args.clients_num
    data_owners = [f"client_{i+1}" for i in range(clients_num)]
    server = "server"
    label_owner = "label_owner"
    model_locations = data_owners + [server, label_owner]

    distributed_testloader = setup_data(config, data_owners)
    
    models = setup_models(data_owners)

    splitnn = SplitNN(models, data_owners, label_owner, server, device, model_locations)
    splitnn.to(device)
    if args.model_training:
        splitnn = models_training(splitnn, distributed_testloader, device, config)
    else:
        load_dir = './best_models'
        load_models(splitnn, load_dir)

    splitnn.eval()
    splitnn.activate_model(device)

    success_num = 0
    attack_num = 0
    count = 0
    constraint = args.constraint
    comb = list(combinations(splitnn.data_owners, constraint))
    warm_round = 50

    cts =Gaussian_MAB_TS(comb, warm_round )
  
    epochs = 3
   
    if args.targeted:
        target_labels = torch.randint(0,9, (1,), device = device)  
    
    
    budget = config['budget']
    
    asr = []
    query_record = []
    count = 0
    query = 0
    success_num = 0
    attack_num = 0
    
    for epoch in range(epochs):
        for data_ptr, labels in tqdm(distributed_testloader):
            lb = 0.0
            ub = 0.0
            attack_obj, indice, competitive = cts.CTS_sample()

            ##random
        #  attack_obj = choice(comb)
        #  indice = comb.index(attack_obj)
          

            for owner in splitnn.data_owners:
                data_ptr[owner] = data_ptr[owner].to(device)
    
            
            batchsize = labels.size(0)
            embedding = splitnn.forward_client(data_ptr)
            for obj in attack_obj:
                lb = torch.minimum(embedding[obj].cpu(), torch.tensor([lb])).min().item()
                ub = torch.maximum(embedding[obj].cpu(), torch.tensor([ub])).max().item()
            epsilon = budget * (ub - lb)
            
            ## attack for a batch data
            if args.targeted:
                target_labels = target_labels
            else:
                target_labels = labels
            result = bandit_nes_adv(embedding, attack_obj,  target_labels, splitnn, lb, ub, epsilon, config, device, args)
            torch.cuda.empty_cache()
            count = count + 1
            query = query + result['average_queries']
            cts.update(indice, result['success_rate'], 1)
            success_num = success_num + result['number_success']
            attack_num = attack_num + batchsize

            if count == args.record_rounds:
                asr.append((100*success_num/attack_num).cpu().item())
                success_num = 0
                attack_num = 0
                query_record.append(query/count)
                count = 0
                query = 0    

    round_record = list(range(1,len(asr)+1))

    # Save results
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%m%d%H%M")
    np.save(os.path.join(results_dir, f'results_{time_stamp}.npy'), [asr, query_record, round_record])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run adversarial attack on Split Neural Network")
    parser.add_argument('--clients_num', type=int, required=True, help='Number of clients')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--constraint', type=int, default=2, help='The maximum corrupted clients')
    parser.add_argument('--record_rounds', type=int, default=125, help='Record the asr every x rounds')
    parser.add_argument('--warm_rounds', type=int, default=50, help='The warm rounds in E-TS Alg')
    parser.add_argument('--model_training', type=str2bool, default=False, help='Whether to train the model (True/False)')
    parser.add_argument('--targeted', type=str2bool, default=True, help='Whether to launch the targeted attack (True/False)')
    args = parser.parse_args()
    
    main(args)