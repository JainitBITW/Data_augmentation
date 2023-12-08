import os
from torch import optim, nn, Tensor , utils
from transformers import GPT2Tokenizer, GPT2LMHeadModel , AutoConfig
import torch
import re 
from tqdm import tqdm
import wandb
COMMIT = "3414f555ca1786e4eb0631b0e2d5c9bc0c6e962faa"
N_STEPS = 50000
batch_size = 8

path = f"/tmp/{os.environ['USER']}/{COMMIT}"
try:
    os.mkdir(path) 
except:
    pass
config= {"lr_scheduler":"CosineAnnealing","commit":COMMIT ,"n_steps":N_STEPS, "random_seed": 42, "batch_size": batch_size, "epochs": 50, "lr": 1e-4, "weight_decay": 1e-4, "eps": 1e-4, "model_name": "gpt2-model-untouched", "dataset": "eng_news_2020_1M-sentences.txt"}

wandb.init(project="gpt2-exp", name = "nn_normal_gpt2",dir=path, config=config)


sentences = []
with open("../data/eng_news_2020_1M-sentences.txt" , "r") as f:
    for line in f:
        sentences.append(line.split("\t")[1].strip())
# sentences = sentences[:1000]


# sentences
def preprocess(input_string):
    # Define a regular expression pattern to match commas between numbers
    pattern = r'(?<=\d)[,/](?=\d)'
    # Use re.sub to replace the matched commas with an empty string
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    result = re.sub(pattern, '', input_string)
    result = re.sub(r'\d+', ' <NUM> ', result)
    result = re.sub(r'-+', '- ', result)
    # add space before and after punctuation
    
    return result

# Apply the remove_commas_between_numbers function to the raw_sentences and take a look at the result
sentences = [preprocess(r) for r in sentences]





sentences = ["<BOS> " + r + " <EOS>" for r in sentences]



# # now replace words with less than 3 occurences with <UNK>
# from collections import Counter

# word_counts = Counter()
# for sentence in sentences:
#     word_counts.update(sentence.split(" "))

# # now replace words with less than 3 occurences with <UNK>
# for i in range(len(sentences)):
#     sentences[i] = " ".join([word if word_counts[word] > 3  else " <UNK> " for word in sentences[i].split(" ")])




tokenizer = GPT2Tokenizer.from_pretrained('../gpt2-model-untouched', bos_token='<BOS>', eos_token='<EOS>', pad_token='<PAD>', unk_token='<UNK>',  special_tokens={"num_token": "<NUM>"})
# tokenizer.save_pretrained('../gpt2-model-untouched')








shuffled_sentences = sentences.copy()
import random
random.seed(42)
random.shuffle(shuffled_sentences)
sentences = shuffled_sentences

train_sentences = sentences[:int(len(sentences)*0.8)]
val_sentences = sentences[int(len(sentences)*0.8):int(len(sentences))]


model = GPT2LMHeadModel.from_pretrained('../gpt2-model-untouched')


optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4 , eps=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id , reduction="none")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0)
# scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
# scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()



def make_batch(sentences):
    tokenised_sentences = tokenizer(sentences, padding=True, truncation=True , max_length=512)
    input_ids = torch.tensor(tokenised_sentences.input_ids).to(device)
    input_ids = input_ids[:, :-1].contiguous()
    attention_mask = torch.tensor(tokenised_sentences.attention_mask).to(device)
    attention_mask = attention_mask[:, :-1].contiguous()
    labels = torch.tensor(tokenised_sentences.input_ids).to(device)
    labels = labels[:, 1:].contiguous()
   
    return input_ids, attention_mask, labels
     

def train_one_batch(model, optimizer, scheduler, criterion, input_ids, attention_mask, labels, steps):
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    loss = criterion(logits.permute(0, 2, 1), labels)
    loss = loss.mean()    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  
    if scheduler is not None:
        scheduler.step()
    wandb.log({"train_loss_per_step": loss.item(), "steps": steps})
    
    return loss.item()


def validate(model, val_sentences, batch_size, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for j in range(0, len(val_sentences), batch_size):
            input_ids, attention_mask, labels = make_batch(val_sentences[j:j + batch_size])
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = criterion(logits.permute(0, 2, 1), labels)
            loss = loss.mean()
            val_loss += loss.item()

    model.train()
    return val_loss / (len(val_sentences) / batch_size)


def train_all(model, optimizer, scheduler, criterion, train_sentences, val_sentences, epochs, batch_size):
    steps = 0
    total_loss = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        # if scheduler is not None:
        #     optimizer.param_groups[0]['lr'] = scheduler.get_last_lr()[0]
        
        bb = 0 
        for i in tqdm(range(0, len(train_sentences), batch_size)):
            wandb.log({"lr":scheduler.get_last_lr()[0] , "steps":steps} )
            # print(type(scheduler.get_last_lr()))
            steps += 1
            bb+=1
            input_ids, attention_mask, labels = make_batch(train_sentences[i:i + batch_size])
            loss_per_batch = train_one_batch(model, optimizer, scheduler, criterion, input_ids, attention_mask, labels, steps)
            total_loss += loss_per_batch

            if steps % 50000 == 0:
                val_loss = validate(model, val_sentences, batch_size, criterion)
                train_loss_now = total_loss / i
                print(f"Val loss: {val_loss}, STEPS: {steps}")
                wandb.log({"val_loss": val_loss, "train_loss_till": train_loss_now,"steps":steps})
                model.save_pretrained(f"/scratch/g2_normal/{steps}")

        print(f"Train loss: {total_loss / bb}")
        wandb.log({"train_loss_per_epoch": total_loss /bb, "epoch": epoch,
                   "total_loss_undivided": total_loss ,"steps":steps})
        total_loss = 0
        bb=0

    model.save_pretrained(f"/scratch/g2_normal/{steps}")

# training the model
epochs = 50
batch_size = 8
steps = 0 
total_loss = 0
train_all(model , optimizer , scheduler , criterion , train_sentences, val_sentences, epochs , batch_size )
