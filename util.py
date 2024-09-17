import torch
from torch import nn
import numpy as np
import pdb
from torchvision import datasets, transforms
from data_dis import Distribute_data
from tqdm import tqdm
import os

def norm(t):
    assert len(t.shape) == 4
    norm_vec = torch.sqrt(t.pow(2).sum(dim=[1,2,3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float()*1e-8
    return norm_vec

def eg_step(x, g, lr):
    real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
    pos = real_x*torch.exp(lr*g)
    neg = (1-real_x)*torch.exp(-lr*g)
    new_x = pos/(pos+neg)
    return new_x*2-1

def linf_step(image, g, lr, attack_obj):
    for obj in attack_obj:
        image[obj] = image[obj] - lr*torch.sign(g[obj]) 
    return image

def nes_step(image, g, lr, attack_obj):
    for obj in attack_obj:
        image[obj] =  image[obj] - lr*torch.sign(g[obj]) 
    return image

def linf_proj(image, eps):
    orig = image.copy()
    def proj(new_x):
        for obj in orig.keys():
            new_x[obj] = orig[obj] + torch.clamp(new_x[obj] - orig[obj], -eps, eps)
        return new_x
    return proj

def dense_to_onehot(y_test, n_cls):
    y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), y_test.cpu()] = True
    return y_test_onehot

def margin_loss(image, noise, attack_object, labels, splitnn, epsilon, lb,ub, device, targeted = True):
    emb = image.copy()
    for obj in attack_object:
        emb[obj] = torch.clamp(emb[obj]+ torch.clamp(noise[obj], -epsilon, epsilon), lb, ub)
    logits = splitnn.forward_server(emb).cpu().detach().numpy()
    y = dense_to_onehot(labels, 10)
    preds_correct_class = (logits * y).sum(1, keepdims=True)
    diff = preds_correct_class - logits  # difference between the correct class and all other classes
    diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
    margin = diff.min(1, keepdims=True)
    loss = margin * -1 if targeted else margin
    return torch.tensor(loss, device = device)

def bandit_nes_adv(embedding, attack_obj, target_label, splitnn,lb, ub, epsilon, config, device, args):
    client = splitnn.data_owners
    max_queries = 2000
    image = embedding.copy()
    orig_images = image.copy()
    batch_size = embedding[attack_obj[0]].size(0)
    total_queries = torch.zeros(batch_size,device =device)
    prior = {}
    zero_noise = {}
    dim = {}
    if args.targeted:
        target_label = target_label.expand(batch_size).to(device)
    else:
        target_label = target_label.view(-1).to(device)
    for obj in attack_obj:
        prior[obj] = torch.zeros_like(embedding[obj]).to(device)
        zero_noise[obj] = torch.zeros_like(embedding[obj]).to(device)
        dim[obj] = prior[obj].nelement()/batch_size

    prior_step = eg_step
    image_step = linf_step
    proj_step = linf_proj(image, epsilon)  

    def function(x, attack_obj, noises = zero_noise):
        with torch.no_grad():
            x_copy = x.copy()
            for obj in attack_obj:
                x_copy[obj] = x_copy[obj] + noises[obj]
            return splitnn.forward_server(x_copy)

    t = 0

    orig_classes = splitnn.forward_server(image).argmax(1)
    if args.targeted:
        correct_classified_mask = (orig_classes != target_label).float()
    else:
        correct_classified_mask = (orig_classes == target_label).float()
    not_dones_mask = correct_classified_mask.clone()


    while not torch.any(total_queries > max_queries):
        t += config['gradient_iters']*2
        if t >= config['max_queries']:
            break
        if not config['nes']:
            ## Updating the prior: 
            # Create noise for exporation, estimate the gradient, and take a PGD step
            exp_noise = {}
            noise1 = {}
            noise2 = {}
            for obj in attack_obj:
                exp_noise[obj] = config['exploration']*torch.randn_like(prior[obj])/(dim[obj]**0.5) 
                q1 = prior[obj] + exp_noise[obj]
                q2 = prior[obj] - exp_noise[obj]
                noise1[obj] = config['fd_eta']*q1/norm(q1)
                noise2[obj] = config['fd_eta']*q2/norm(q2)
         
            l1 = margin_loss(image, noise1, attack_obj, target_label, splitnn, epsilon,lb,ub,device,targeted = args.targeted)
            l2 =  margin_loss(image, noise2, attack_obj, target_label, splitnn, epsilon,lb,ub,device, targeted = args.targeted)
    
   
            est_deriv = (l1 - l2)/(config['fd_eta']*config['exploration'])
        
            est_grad = {}
            for obj in attack_obj:
                est_grad[obj] = est_deriv.view(-1, 1, 1, 1)*exp_noise[obj]
                prior[obj] = prior_step(prior[obj], est_grad[obj], config['online_lr'])
        else:
            prior = {}
            for obj in attack_obj:
                prior[obj]= torch.zeros_like(image[obj])
            for _ in range(config['gradient_iters']):
                exp_noise = {}
                noise = {}
                neg_noise = {}
                for obj in attack_obj:
                    exp_noise[obj] = torch.empty_like(image[obj]) 
                    nn.init.normal_(exp_noise[obj])
                    noise[obj] = config['fd_eta']*exp_noise[obj]
                    neg_noise[obj] = -config['fd_eta']*exp_noise[obj]
                pos_g = margin_loss(image, noise, attack_obj, target_label, splitnn, epsilon,lb,ub,device, targeted = args.targeted)
                neg_g = margin_loss(image, neg_noise, attack_obj, target_label, splitnn,epsilon,lb,ub,device, targeted = args.targeted)
                for obj in attack_obj:
                    prior[obj] += pos_g.view(-1, 1, 1, 1)*exp_noise[obj]
                    prior[obj] -= neg_g.view(-1, 1, 1, 1)*exp_noise[obj]
            for obj in attack_obj:
                prior[obj] = prior[obj] / (2* config['gradient_iters']* config['fd_eta'])
        
        correct_prior = {}
        for obj in attack_obj:           
            correct_prior[obj] = prior[obj]*correct_classified_mask.view(-1, 1, 1, 1).cuda()
        
        if not config['nes']:            
            image_step(image, correct_prior, config['image_lr'], attack_obj)
        else:
            nes_step(image, correct_prior, config['image_lr'], attack_obj)
        proj_step(image)
        for cl in client:
            image[cl] = torch.clamp(image[cl], lb, ub)
            if not (image[cl] - orig_images[cl]).max() <= epsilon + 1e-3:
                pdb.set_trace()

        
        ## Continue query count
        total_queries += 2*config['gradient_iters']* not_dones_mask
        if args.targeted:
            not_dones_mask = not_dones_mask*((function(image, attack_obj = attack_obj).argmax(1) != target_label).float())
        else:
            not_dones_mask = not_dones_mask*((function(image, attack_obj = attack_obj).argmax(1) == target_label).float())
        ## Logging stuff
        success_mask = (1 - not_dones_mask)*correct_classified_mask
        num_success = success_mask.sum()
        current_success_rate = (num_success/correct_classified_mask.sum()).cpu().item()
        success_queries = ((success_mask*total_queries).sum()/(num_success + 1e-6)).cpu().item()
        max_curr_queries = total_queries.max().cpu().item()
      #  if args.log_progress:
       #     print("Queries: %d | Success rate: %f | Average queries: %f" % (max_curr_queries, current_success_rate, success_queries))

        if current_success_rate == 1.0:
            break

    return {
            'average_queries': success_queries,
            'num_correctly_classified': correct_classified_mask.sum().cpu().item(),
            'success_rate': current_success_rate,
            'number_success':num_success,
            #'images_orig': orig_images,
            #'images_adv': image,
            #'all_queries': total_queries.cpu().numpy(),
            # 'correctly_classified': correct_classified_mask.cpu().numpy(),
           #'success_mask': success_mask.cpu().numpy()
    }

def models_training(splitnn, testloader,device, config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    distributed_trainloader = Distribute_data(data_owners=splitnn.data_owners, data_loader=trainloader)
    
    splitnn.activate_model(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(config['training_epochs']):
        splitnn.train()
        running_loss = 0
        
        # Training loop
        progress_bar = tqdm(distributed_trainloader, desc=f"Epoch {epoch+1}/{config['training_epochs']}")
        for data, labels in progress_bar:
            for owner in splitnn.data_owners:
                data[owner] = data[owner].to(device)
            labels = labels.to(device)
            
            splitnn.zero_grads()
            pred = splitnn.forward(data)
            loss = criterion(pred, labels)
            loss.backward()
            splitnn.steps()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Evaluation
        accuracy = evaluate(splitnn, testloader, device)
        
        print(f"Epoch {epoch+1}/{config['training_epochs']}")
        print(f"Training Loss: {running_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_models(splitnn, './best_models')
    
    print(f"Best Test Accuracy: {best_accuracy:.4f}")
    return splitnn

def evaluate(splitnn, dataloader, device):
    splitnn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in dataloader:
            for owner in splitnn.data_owners:
                data[owner] = data[owner].to(device)
            label = label.to(device)
            outputs = splitnn.forward(data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return correct / total

def save_models(splitnn, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for location, model in splitnn.models.items():
        torch.save(model.state_dict(), os.path.join(save_dir, f'{location}_model.pth'))
    print(f"Models saved in {save_dir}")





class Gaussian_MAB_TS():
    def __init__(self, combination, warm_round):
        self.combination = combination
        self.upper = torch.tensor([0 for _ in range(len(combination))])
        self.emp = torch.tensor([1.0 for _ in range(len(combination))])
        self.round = 0
        self.choice_num = torch.tensor([0 for _ in range(len(combination))])
        self.warm_round = warm_round
        self.mean = torch.tensor([0 for _ in range(len(combination))], dtype = torch.float32)
        self.std = torch.tensor([1.0 for _ in range(len(combination))], dtype = torch.float32)

    def CTS_sample(self):
        self.round += 1
        emp_mask = (self.choice_num >= self.round/len(self.combination))
        sample_mask = torch.where(emp_mask == True, 1, 0)
        max_mu,k_max = torch.max(torch.mul(sample_mask, self.mean),0)
        competitive = self.emp >= max_mu
        competitive[k_max] = True
        competitive = torch.where(competitive == True, 1, 0)

        if self.round > self.warm_round:
            sample = torch.normal(self.mean, self.std)
            sample = torch.mul(sample, competitive)
            indice = torch.max(sample,0)[1]
            attack_obj = self.combination[indice]
        else:
            sample = torch.normal(self.mean, self.std)
            indice = torch.max(sample,0)[1]
            attack_obj = self.combination[indice]
            
        return attack_obj, indice, competitive

    def update(self, indice, grad, batchsize):
        self.choice_num[indice] = self.choice_num[indice] + batchsize
        self.upper[indice] = self.upper[indice].item() if self.upper[indice] >= grad else grad
        self.emp[indice] = (self.emp[indice] * (self.choice_num[indice]-1) + self.upper[indice])/(self.choice_num[indice])
        self.mean[indice] = (self.mean[indice]* (self.choice_num[indice]-1) + grad)/ (self.choice_num[indice])
        self.std[indice] = 1 / (self.choice_num[indice] + 1)

