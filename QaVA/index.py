import torch
import torchvision
import QaVA
import numpy as np
from tqdm.autonotebook import tqdm
import pickle
import os, shutil
from torch.utils.tensorboard import SummaryWriter
from QaVA.utils import propagate, suppress_stdout_stderr
from QaVA.model import *
from QaVA.losses import *
from QaVA.data import ScoreDataset
import math

class Index:
    def __init__(self, config):
        self.config = config
        if self.config.do_training:
            self.target_dnn_cache = QaVA.DNNOutputCache(
                self.get_target_dnn(),
                self.get_target_dnn_dataset(train_or_test='test'),
                self.target_dnn_callback
            )
            self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='test')
        else:
            self.target_dnn_cache = None
        self.seed = self.config.seed
        self.rand = np.random.RandomState(seed=self.seed)
        
    def override_target_dnn_cache(self, target_dnn_cache, train_or_test='train'):
        '''
        This allows you to override score_net.utils.DNNOutputCache if you already have the target dnn
        outputs available somewhere. Returning a list or another 1-D indexable element will work.
        '''
        return target_dnn_cache
        
    def score_fn(self, target_dnn_output):
        '''
        Define your notion of "closeness" as described in the paper between records "a" and "b".
        Return a Boolean.
        '''
        raise NotImplementedError
        
    def get_target_dnn_dataset(self, train_or_test='train'):
        '''
        Define your target_dnn_dataset under the condition of "train_or_test".
        Return a torch.utils.data.Dataset object.
        '''
        raise NotImplementedError
    
    def get_embedding_dnn_dataset(self, train_or_test='train'):
        '''
        Define your embedding_dnn_dataset under the condition of "train_or_test".
        Return a torch.utils.data.Dataset object.
        '''
        raise NotImplementedError
        
    def get_target_dnn(self):
        '''
        Define your Target DNN.
        Return a torch.nn.Module.
        '''
        raise NotImplementedError
        
    def get_score_dnn(self):
        '''
        Define your Score DNN.
        Return a torch.nn.Module.
        '''
        raise NotImplementedError
        
    def get_pretrained_embedding_dnn(self):
        '''
        Define your Embeding DNN.
        Return a torch.nn.Module.
        '''
        return self.get_pretrained_embedding_dnn()
        
    def target_dnn_callback(self, target_dnn_output):
        '''
        Often times, you want to process the output of your target dnn into something nicer.
        This function is called everytime a target dnn output is computed and allows you to process it.
        If it is not defined, it will simply return the input.
        '''
        return target_dnn_output
    
    def do_offline_indexing(self):
        print('[Stage] Offline Indexing')
        if self.config.do_offline_indexing:
            model = self.get_pretrained_embedding_dnn()
            try:
                model.cuda()
                model.eval()
            except:
                pass

            dataset = self.get_embedding_dnn_dataset(train_or_test='test')
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0, # not support
                pin_memory=False
            )
            
            embeddings = []
            for batch in tqdm(dataloader, desc='Embedding DNN'):
                batch = batch.cuda()
                with torch.no_grad():
                    output = model(batch).cpu()
                embeddings.append(output)
            del dataloader
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.numpy()
            self.training_embeds = embeddings.astype(np.float32)
            np.save(os.path.join(self.config.cache_root, 'training_embeds.npy'), self.training_embeds)
            print('[Offline Indexing] Save training_embeds.npy')
        else:
            try:
                self.training_embeds = np.load(os.path.join(self.config.cache_root, 'training_embeds.npy')).astype(np.float32)
                print('[Offline Indexing] Use cached training_embeds.npy')
            except:
                raise RuntimeError('[Offline Indexing] Cannot find training_embeds.npy')

    def do_mining(self):
        print('[Stage] Index Mining')
        if self.config.do_mining:
            bucketter = QaVA.bucketters.FPFRandomBucketter(self.config.nb_train, self.seed)
            self.training_idxs, self.topk_reps, self.topk_dists = bucketter.bucket(self.training_embeds, self.config.max_k)
            np.save(os.path.join(self.config.cache_root, 'training_idxs.npy'), self.training_idxs)
            np.save(os.path.join(self.config.cache_root, 'topk_reps.npy'), self.topk_reps)
            np.save(os.path.join(self.config.cache_root, 'topk_dists.npy'), self.topk_dists)
            print('[Mining] Save training_idxs.npy, topk_reps.npy and topk_dists.npy')
        else:
            try:
                self.training_idxs = np.load(os.path.join(self.config.cache_root, 'training_idxs.npy'))
                self.topk_reps = np.load(os.path.join(self.config.cache_root, 'topk_reps.npy'))
                self.topk_dists = np.load(os.path.join(self.config.cache_root, 'topk_dists.npy'))
                print('[Mining] Use cached training_idxs.npy, topk_reps.npy and topk_dists.npy')
            except:
                raise RuntimeError('[Mining] Cannot find training_idxs.npy, topk_reps.npy and topk_dists.npy')
            
    def do_propagation(self):
        print('[Stage] Propagation')
        if self.config.do_propagation:
            for idx in tqdm(self.training_idxs, desc='Target DNN'):
                self.target_dnn_cache[idx]
            self.prop_preds, self.repr_labels = propagate(self.target_dnn_cache, self.training_idxs, self.topk_reps, self.topk_dists, self.score_fn)
            np.save(os.path.join(self.config.cache_root, 'prop_preds.npy'), self.prop_preds)
            np.save(os.path.join(self.config.cache_root, 'repr_labels.npy'), self.repr_labels)
            print('[Propagation] Save prop_preds.npy and repr_labels.npy')
        else:
            try:
                self.prop_preds = np.load(os.path.join(self.config.cache_root, 'prop_preds.npy'))
                self.repr_labels = np.load(os.path.join(self.config.cache_root, 'repr_labels.npy'))
                print('[Propagation] Use cached prop_preds.npy and repr_labels.npy')
            except:
                raise RuntimeError('[Propagation] Cannot find prop_preds.npy and repr_labels.npy')
    
    def do_training(self, save_cache=True, finetune=False):
        print('[Stage] Index Training')
        self.unique_labels, self.repr_label_idx = np.unique(self.repr_labels.astype(float).astype(int), return_inverse=True)
        if self.config.do_training:
            def shuffle(x, seed=0):
                rand = np.random.RandomState(seed=seed)
                rand.shuffle(x)
                return x
            self.training_idxs = shuffle(self.training_idxs)
            self.repr_labels = shuffle(self.repr_labels)
            self.repr_label_idx = shuffle(self.repr_label_idx)

            n_val = int(len(self.training_idxs) * self.config.val_rate)
            train_idxs, val_idxs = self.training_idxs[n_val:], self.training_idxs[:n_val]
            train_labels, val_labels = self.repr_labels[n_val:], self.repr_labels[:n_val]
            train_label_idx, val_label_idx = self.repr_label_idx[n_val:], self.repr_label_idx[:n_val]
            
            if finetune:
                model = self.score_dnn
            else:
                model = self.get_score_dnn()
            model.train()
            model.cuda()

            is_classify = isinstance(model, ClassifyNet)
            train_dataset = ScoreDataset(
                embeddings=self.training_embeds,
                preds=self.prop_preds,
                list_of_idxs=train_idxs,
                labels=train_labels,
                label_idx=train_label_idx,
                is_classify=is_classify
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0, # not support
                pin_memory=False
            )

            val_dataset = ScoreDataset(
                embeddings=self.training_embeds,
                preds=self.prop_preds,
                list_of_idxs=val_idxs,
                labels=val_labels,
                label_idx=val_label_idx,
                is_classify=is_classify
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=int(self.config.batch_size * self.config.val_rate),
                shuffle=True,
                num_workers=0, # not support
                pin_memory=False
            )

            #mask = torch.ones(len(self.unique_labels))
            #mask[2:] = 0
            mask = None
            if is_classify:
                loss_fn = torch.nn.CrossEntropyLoss()
                #loss_fn = ReweightedCrossEntropyLoss(labels=self.repr_labels, unique_labels=self.unique_labels, mask=mask)
                #loss_fn = FocalLoss(labels=self.repr_labels, unique_labels=self.unique_labels, gamma=2, mask=mask)
            else:
                #loss_fn = torch.nn.MSELoss()
                #loss_fn = torch.nn.L1Loss() # MAE
                #loss_fn = torch.nn.HuberLoss()
                loss_fn = HuberLoss(labels=self.repr_labels, unique_labels=self.unique_labels, weight_fn=self.config.weight_fn, mask=mask)
            try:
                loss_fn = loss_fn.cuda()
            except:
                pass
        
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.train_lr)
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
            
            tb_path = os.path.join(self.config.cache_root, 'logger')
            if os.path.exists(tb_path):
                shutil.rmtree(tb_path)
            os.makedirs(tb_path)
            writer = SummaryWriter(tb_path)

            def _run(coord, embed, label, prop_pred):
                coord, embed = coord.cuda(), embed.cuda()
                label = label.cuda()
                if isinstance(model, TimeNet):
                    pred = model(coord[:, :1], embed)
                    if isinstance(model, ScoreNet):
                        pred = pred.view(-1)
                else: # ConditionalNet
                    prop_pred = prop_pred.cuda().to(torch.float32)
                    pred = model(coord, embed, prop_pred)
                return loss_fn(pred, label)
            
            def train():
                avg_loss = 0
                for coord, embed, label, prop_pred in train_dataloader:
                    loss = _run(coord, embed, label, prop_pred)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss
                return avg_loss / len(train_dataloader)
            
            def val():
                avg_loss = 0
                for coord, embed, label, prop_pred in val_dataloader:
                    with torch.no_grad():
                        loss = _run(coord, embed, label, prop_pred)
                    avg_loss += loss
                return avg_loss / len(val_dataloader)
            
            min_val_loss = np.inf
            early_stop_cnt = 0
            best_model = model
            for i in range(self.config.max_training_epochs):
                loss = train()
                val_loss = val()
                print(f'[Training] epoch={i+1:05d} lr={optimizer.state_dict()["param_groups"][0]["lr"]:.0e} train={loss:.6f}, val={val_loss:.6f}')
                writer.add_scalar('Train Loss', loss, i)
                writer.add_scalar('Val Loss', val_loss, i)

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    early_stop_cnt = 0
                    best_model = model
                    if save_cache:
                        torch.save(best_model.state_dict(), os.path.join(self.config.cache_root, 'model.pt'))
                    print(f'[Training] Use model.pt at epoch-{i} (val={val_loss:.6f})')
                if i > 5 and (val_loss - loss) / val_loss > 0.2:
                    early_stop_cnt += 1
                if early_stop_cnt >= self.config.early_stop:
                    print(f'[Training] Stop at epoch-{i}')
                    break
                #scheduler.step(val_loss)
            self.score_dnn = best_model
        else:
            model = self.get_score_dnn()
            model.cuda()
            try:
                checkpoint = torch.load(os.path.join(self.config.cache_root, 'model.pt'))
                model.load_state_dict(checkpoint)
                self.score_dnn = model
                print('[Training] Use cached model.pt')
            except:
                raise RuntimeError('[Training] Cannot find model.pt')
            
    def do_infer(self, save_cache=True):
        print('[Stage] Index Inferring')
        del self.target_dnn_cache
        self.target_dnn_cache = QaVA.DNNOutputCache(
            self.get_target_dnn(),
            self.get_target_dnn_dataset(train_or_test='test'),
            self.target_dnn_callback
        )
        self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='test')

        if self.config.do_infer:
            model = self.score_dnn
            try:
                model.eval()
                model.cuda()
            except:
                pass

            n = self.training_embeds.shape[0]
            batch_size = self.config.batch_size
            t_coords = torch.linspace(0, 1, n)
            pred_coords = torch.from_numpy((self.prop_preds - self.prop_preds.min()) / (self.prop_preds.max() - self.prop_preds.min()))
            #pred_coords = torch.from_numpy(self.prop_preds)
            coords = torch.stack((t_coords, pred_coords), dim=-1)
            scores = []
            for i in tqdm(range(math.ceil(n / batch_size)), desc='Inference'):
                embed = self.training_embeds[i*batch_size: min((i+1)*batch_size, n)]
                embed = torch.from_numpy(embed).float()
                coord = coords[i*batch_size: min((i+1)*batch_size, n)]
                embed = embed.cuda()
                coord = coord.cuda()
                with torch.no_grad():
                    if isinstance(model, TimeNet):
                        score = model.get_score(coord[:, :1], embed).cpu()
                    else: # ConditionalNet
                        prop_pred = self.prop_preds[i*batch_size: min((i+1)*batch_size, n)]
                        prop_pred = torch.from_numpy(prop_pred).float().cuda()
                        score = model.get_score(coord, embed, prop_pred).cpu()
                score = torch.clip(score, min=0)
                scores.append(score)
            scores = torch.cat(scores, dim=0)
            scores = scores.cpu().numpy().astype(np.float32)

            # set real values for repr
            # for idx in self.training_idxs:
            #     scores[idx] = float(score_net.DNNOutputCacheFloat(
            #         self.target_dnn_cache, self.score_fn, idx
            #     ))

            if save_cache:
                np.save(os.path.join(self.config.cache_root, 'scores.npy'), scores)
                print('[Inferring] Save scores.npy')
            
            self.scores = scores
        elif os.path.exists(os.path.join(self.config.cache_root, 'scores.npy')):
            print('[Inferring] Use cached scores.npy')
            self.scores = np.load(os.path.join(self.config.cache_root, 'scores.npy')).astype(np.float32)
        else:
            raise RuntimeError('[Inferring] Cannot find scores.npy')

    def init(self):
        self.do_offline_indexing()
        torch.cuda.empty_cache()
        self.do_mining()
        torch.cuda.empty_cache()
        self.do_propagation()
        torch.cuda.empty_cache()
        self.do_training()
        torch.cuda.empty_cache()
        self.do_infer()
        torch.cuda.empty_cache()

    def update(self, training_idxs, training_labels, finetune, prob_preds=None, lr=None, val_rate=None, logging=False):
        self.training_idxs = training_idxs
        self.repr_labels = training_labels
        if prob_preds is not None:
            self.prob_preds = prob_preds
        self.config.do_training = True
        self.config.do_infer = True
        if lr is not None:
            self.config.lr = lr
        if val_rate is not None:
            self.config.val_rate = val_rate
        
        if logging:
            self.do_training(save_cache=False, finetune=finetune)
            torch.cuda.empty_cache()
            self.do_infer(save_cache=False)
        else:
            with suppress_stdout_stderr():
                self.do_training(save_cache=False, finetune=finetune)
            torch.cuda.empty_cache()
            with suppress_stdout_stderr():
                self.do_infer(save_cache=False)
        torch.cuda.empty_cache()