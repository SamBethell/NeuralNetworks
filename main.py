import torch as t
import torch.nn as nn
import json
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import torchvision
import numpy as np
from torchvision import datasets, transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as f
import torch.optim
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from transformers import AutoTokenizer, AutoModel


# Making cnn to train lstm + nn to read questions and answers from a page of notes
######################################################################################################################################

class CustomDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.cache = {}
        self.valid_pairs = self._build_pairs()

    def _build_pairs(self):
        pairs = []
        for img_file in sorted(os.listdir(self.img_dir)):
            base_name = os.path.splitext(img_file)[0]
            annotation_file = f"{base_name}.json"
            img_path = os.path.join(self.img_dir, img_file)
            annotation_path = os.path.join(self.annotation_dir, annotation_file)
            if not os.path.exists(annotation_path):
               continue
            try:
                with open(annotation_path, 'r') as f:
                    json.load(f)
                Image.open(img_path).convert('RGB')
                pairs.append((img_path, annotation_path))
            except:
                print('Skipping invalid pair')
        return pairs

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            img_path, annotation_path = self.valid_pairs[idx]
            image = Image.open(img_path).convert('RGB')
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)

            image = self.transform(image)
            self.cache[idx] = (image, annotation)
            return image, annotation

    def get_qa_pairs(self, dictionary):
        qa_pairs = []
        items = {item['id']:item for item in dictionary['form']}
        for item_id, item in items.items():
            if item['label'] == 'question':
                x1, y1, x2, y2 = item['words'][0]['box']
                for link in item['linking']:
                    if len(link) == 2:
                        answer_id = link[1]
                        if answer_id in items and items[answer_id]['label'] == 'answer':
                            x3, y3, x4, y4 = items[answer_id]['words'][0]['box']
                            if x4 > x1 and y4 > y1:
                                qa_box = [x1, x4, y1, y4]
                                qa_pairs.append({'question': item['text'], 'answer': items[answer_id]['text'], 'box':qa_box})
                            else:
                                continue
        return qa_pairs

def custom_collate(batch):
    '''
    To deal with inconsistent batch sizes
    '''
    if len(batch) != 5:
        num_missing = 5 - len(batch)
        batch += batch[:num_missing]

    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]
    images = torch.stack(images)
    return images, annotations

def compute_embeddings(qa_pairs, tokenizer, model, device):
    embeddings = {}
    with torch.no_grad():
        for item in qa_pairs:
            for i in range(len(item)):
                text = item[i]['question']+''+item[i]['answer']
                if text not in embeddings:
                    inputs = tokenizer(text, return_tensors='pt', truncation=True).to(device)
                    outputs= model(**inputs)
                    embeddings[text] = outputs.last_hidden_state.mean(dim=1)
    return embeddings

transform = transforms.Compose([
    transforms.Resize((777, 1000)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomDataset(
    img_dir="C:/Users/samue/python/AutoFlashcards/dataset/training_data/images/",
    annotation_dir="C:/Users/samue/python/AutoFlashcards/dataset/training_data/annotations",
    transform=transform
)

dataset_test = CustomDataset(
    img_dir="C:/Users/samue/python/AutoFlashcards/dataset/testing_data/images/",
    annotation_dir="C:/Users/samue/python/AutoFlashcards/dataset/testing_data/annotations/",
    transform=transform
)

# Create DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=custom_collate)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=5, shuffle=True, collate_fn=custom_collate)
class RCNN(nn.Module):
    def __init__(self, num_channels, num_classes, lstm_hidden_size=256, num_lstm_layers=2):
        super(RCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 256, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True))
        self.lstm = nn.Sequential(nn.LSTM(512, lstm_hidden_size*2, num_lstm_layers, bidirectional=True, batch_first=True))
        self.fc = nn.Linear(lstm_hidden_size*4, num_classes)

    def forward(self, x):
        lstm_input = self.cnn(x) #[B, 512, 1, W]
        lstm_input = lstm_input.squeeze(0) # [B, 512, W]
        lstm_input = lstm_input.permute(2, 1, 0) # [W, B, 512]
        fc_input = self.lstm(lstm_input)
        output = self.fc(fc_input[0])
        return output


def get_by_id(data, target_id):
    return [item for item in data['form'] if item.get('id') == target_id]

def training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rcnn = RCNN(3, 384).to(device)
    criterion = nn.CosineSimilarity().to(device)
    optimizer = torch.optim.Adam(rcnn.parameters(), lr=0.01)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

    # freeze text model
    for param in text_model.parameters():
        param.requires_grad = False

    for epoch in range(1): # only one due to long run time
        rcnn.train()
        running_loss = 0
        total_pairs = 0

        for batch_idx, (images, annotations) in enumerate(dataloader):
            images = images.to(device)
            qa_pairs = []
            img_idx = 0
            current_count = 0
            # process all annotations in batch
            for ann_idx, ann in enumerate(annotations):
                current_count += len(dataset.get_qa_pairs(ann))
                qa_pairs.extend(dataset.get_qa_pairs(ann))
            # precompute all embeddings for this batch
            embeddings = {}
            with torch.no_grad():
                for pair in qa_pairs:
                    text = pair['question'] + ' ' + pair['answer']
                    if text not in embeddings:
                        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
                        outputs = text_model(**inputs)
                        embeddings[text] = outputs.last_hidden_state.mean(dim=1)
            # process each QA pair
            optimizer.zero_grad()

            for pair_idx, pair in enumerate(qa_pairs):
                if pair_idx > current_count:
                    img_idx += 1
                try:
                    box = pair['box']
                    img_region = images[img_idx:img_idx+1, :, box[2]:box[3], box[0]:box[1]]
                    # forward pass
                    output = rcnn(img_region)
                    text = pair['question'] + ' ' + pair['answer']
                    loss = 1 - criterion(output, embeddings[text]).mean()
                    # backward pass
                    loss.backward()
                    running_loss += loss.item()
                    total_pairs += 1
                    # print progress
                    print(
                        f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}/{len(dataloader)}, Pair: {pair_idx + 1}/{len(qa_pairs)}")
                except Exception as e:
                    print(f"Skipping pair due to error: {e}")
                    continue

            # update weights
            optimizer.step()
    return rcnn, tokenizer, device, text_model

def setup_nearest_neighbors(ground_truth_embeddings):
    nn = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn.fit(ground_truth_embeddings)
    return nn

def true_embeddings(dataset, tokenizer, model, device):
    all_text = []
    all_embeddings = []

    for idx in range(len(dataset)):
        image, annotations = dataset[idx]
        qa_pairs = dataset.get_qa_pairs(annotations)
        for pair in qa_pairs:
            text = pair['question']+''+pair['answer']
            if text not in all_text:
                all_text.append(text)
                inputs = tokenizer(text, return_tensors='pt', truncation=True).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(embedding)
    return all_text, np.vstack(all_embeddings)

def test(tokenizer, rcnn, nn_model, device):
    rcnn.eval()
    correct = 0
    total_pairs = 0
    for batch_idx, (image, annotations) in enumerate(dataloader_test):
        img_idx = 0
        current_count = 0
        images = image.to(device)
        qa_pairs =  []
        for ann in annotations:
            qa_pairs.extend(dataset.get_qa_pairs(ann))
            current_count += len(dataset.get_qa_pairs(ann))
        for pair_idx, pair in enumerate(qa_pairs):
            if pair_idx > current_count:
                img_idx += 1
            try:
                box = pair['box']
                img_region = images[img_idx:img_idx+1,:, box[2]:box[3], box[0]:box[1]]

                with torch.no_grad():
                    embeddings = rcnn(img_region).cpu().numpy()
                distances, indices = nn_model.kneighbors(embeddings)
                predicted_text = all_text[indices[0][0]]
                true_text = pair['question'] + ' ' + pair['answer']
            except:
                print("poo")
                continue
            # Check if match is correct
            if predicted_text == true_text:
                correct += 1
            total_pairs += 1

            print(f"Predicted: {predicted_text}")
            print(f"True: {true_text}")
            print(f"Cosine Distance: {distances[0][0]}")


rcnn, tokenizer, device, text_model = training()
all_text, embeddings = true_embeddings(dataset_test, tokenizer, text_model, device)
nn_model = setup_nearest_neighbors(embeddings)
test(tokenizer, rcnn, nn_model, device)


######################################################################################################################################






