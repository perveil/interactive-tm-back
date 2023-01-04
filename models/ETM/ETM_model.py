import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from models.ETM.VAE_model import VAE


class EVAE(VAE):
    def __init__(self, encode_dims, decode_dims, dropout, emb_dim):
        super(EVAE, self).__init__(encode_dims=encode_dims, decode_dims=decode_dims, dropout=dropout)
        self.emb_dim = emb_dim
        self.vocab_size = encode_dims[0]
        self.n_topic = encode_dims[-1]
        self.rho = nn.Linear(emb_dim, self.vocab_size)
        self.alpha = nn.Linear(emb_dim, self.n_topic)
        self.decoder = None

    def decode(self, z):
        wght_dec = self.alpha(self.rho.weight)
        beta = F.softmax(wght_dec, dim=0).transpose(1, 0)
        res = torch.mm(z, beta)
        logits = torch.log(res + 1e-6)
        return logits


class ETM:
    def __init__(self, vocab, bow_dim, n_topic, task_name=None, device=None, emb_dim=300):
        self.vocab = vocab
        self.bow_dim = bow_dim
        self.n_topic = n_topic
        self.emb_dim = emb_dim
        self.vae = EVAE(
            encode_dims=[bow_dim, 1024, 512, n_topic],
            decode_dims=[n_topic, 512, bow_dim],
            dropout=0.0,
            emb_dim=emb_dim
        )
        self.device = device
        self.task_name = task_name
        if device is not None:
            self.vae = self.vae.to(device)

    def train(self, train_data, test_data=None, batch_size=256, learning_rate=1e-3, num_epochs=100, log_every=5,
              beta=1.0, criterion='cross_entropy', ckpt=None):
        self.vae.train()
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)

        if ckpt:
            self.load_model(ckpt)
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
        else:
            start_epoch = 0

        trainloss_lst, valloss_lst = [], []
        for epoch in range(start_epoch, num_epochs):
            epochloss_lst = []
            for iter, data in enumerate(data_loader):
                optimizer.zero_grad()

                bows = data
                bows = bows.to(self.device)
                bows_recon, mus, log_vars = self.vae(bows, lambda x: torch.softmax(x, dim=1))
                if criterion == 'cross_entropy':
                    logsoftmax = torch.log_softmax(bows_recon, dim=1)
                    rec_loss = -1.0 * torch.sum(bows * logsoftmax)
                elif criterion == 'bce_softmax':
                    rec_loss = F.binary_cross_entropy(torch.softmax(bows_recon, dim=1), bows, reduction='sum')
                elif criterion == 'bce_sigmoid':
                    rec_loss = F.binary_cross_entropy(torch.sigmoid(bows_recon), bows, reduction='sum')
                else:
                    raise Exception("Unknown criterion")

                kl_div = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())
                loss = rec_loss + kl_div * beta
                loss.backward()
                optimizer.step()

                trainloss_lst.append(loss.item() / len(bows))
                epochloss_lst.append(loss.item() / len(bows))
                # if (iter + 1) % 2 == 0:
                if True:
                    print(
                        f'Epoch {(epoch + 1):>3d}\tIter {(iter + 1):>4d}\tLoss:{loss.item() / len(bows):<.7f}\t'
                        f'Rec Loss:{rec_loss.item() / len(bows):<.7f}\tKL Div:{kl_div.item() / len(bows):<.7f}')
            if (epoch + 1) % log_every == 0:
                save_name = f'./models/ETM/ckpt/ETM_{self.task_name}_tp{self.n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_ep{epoch + 1}.ckpt'
                checkpoint = {
                    "net": self.vae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "param": {
                        "bow_dim": self.bow_dim,
                        "n_topic": self.n_topic,
                        "task_name": self.task_name,
                        "emb_dim": self.emb_dim,
                        "device": self.device,
                    }
                }
                torch.save(checkpoint, save_name)
                print(f'Epoch {(epoch + 1):>3d}\tLoss:{sum(epochloss_lst) / len(epochloss_lst):<.7f}')
                print(f'Top 10 words for each topic:')
                print('\n'.join([str(lst) for lst in self.show_topic_words(top_k=10)]))
        # model.evaluate(test_data=test_data)
        checkpoint = {
            "net": self.vae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": -1,
            "param": {
                "bow_dim": self.bow_dim,
                "n_topic": self.n_topic,
                "task_name": self.task_name,
                "emb_dim": self.emb_dim,
                "device": self.device,
            }
        }
        save_name = f'./models/ETM/ckpt/ETM_{self.task_name}_tp{self.n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'
        torch.save(checkpoint, save_name)
        print(f"Train end. Model is saved to {save_name}")

    def evaluate(self, test_data, calc4each=False):
        pass
        # topic_words = self.show_topic_words()
        # return evaluate_topic_quality(topic_words, test_data, taskname=self.task_name, calc4each=calc4each)

    def get_topic_word_dist(self):
        self.vae.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topic).to(self.device)
            word_dist = self.vae.decode(idxes)  # word_dist shape: [n_topic, vocab.size]
            word_dist = torch.softmax(word_dist, dim=1)
            return word_dist.detach().cpu().numpy()

    def get_doc_topic_dist(self, train_data):
        self.vae.eval()
        data_loader = DataLoader(train_data, batch_size=512, shuffle=False, num_workers=4)
        embed_lst = []
        for data_batch in data_loader:
            bows = data_batch
            embed = self.inference_by_bow(bows)
            embed_lst.append(embed)
        embed_lst = np.concatenate(embed_lst, axis=0)
        return embed_lst

    def load_model(self, ckpt):
        self.vae.load_state_dict(ckpt["net"])
        self.bow_dim = ckpt["param"]["bow_dim"]
        self.n_topic = ckpt["param"]["n_topic"]
        self.task_name = ckpt["param"]["task_name"]
        self.emb_dim = ckpt["param"]["emb_dim"]
        self.device = ckpt["param"]["device"]

    def show_topic_words(self, topic_id=None, top_k=10):
        topic_words = []
        idxes = torch.eye(self.n_topic).to(self.device)
        word_dist = self.vae.decode(idxes)
        word_dist = torch.softmax(word_dist, dim=1)
        vals, indices = torch.topk(word_dist, top_k, dim=1)
        indices = indices.cpu().tolist()
        if topic_id is None:
            for i in range(self.n_topic):
                topic_words.append([self.vocab[idx] for idx in indices[i]])
        else:
            topic_words.append([self.vocab[idx] for idx in indices[topic_id]])
        return topic_words

    def inference_by_bow(self, doc_bow):
        # doc_bow: torch.tensor [vocab_size]; optional: np.array [vocab_size]
        if isinstance(doc_bow, np.ndarray):
            doc_bow = torch.from_numpy(doc_bow)
        doc_bow = doc_bow.reshape(-1, self.bow_dim).to(self.device)
        with torch.no_grad():
            mu, log_var = self.vae.encode(doc_bow)
            mu = self.vae.fc1(mu)
            theta = F.softmax(mu, dim=1)
            return theta.detach().cpu().squeeze(0).numpy()

