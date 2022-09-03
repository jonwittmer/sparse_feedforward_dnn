import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import prune
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

from save_model_manual import saveSparseModel

gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(torch.cuda.device_count())
if len(gpus) == 0:
    print("using CPU")
    device = "cpu"
elif len(gpus) == 1:
    print("Using GPU")
    device = "cuda" #gpus[0]
    print(device)
else:
    print("Only training with single GPU")
    device = gpus[0]
    
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latentSpace = self.encoder(x)
        return self.decoder(latentSpace)

class AutoencoderWithLatent(Autoencoder):
    def forward(self, x):
        if not (type(x) is tuple or type(x) is list):
            raise ValueError("AutoencoderWithLatent must have input that is a tuple of tensors")
        latentPred = self.encoder(x[0])
        pred = self.decoder(latentPred)
        return pred, latentPred
    
class Encoder(nn.Module):
    def __init__(self, latentDim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linearReluStack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latentDim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        latent = self.linearReluStack(x)
        return latent

class Decoder(nn.Module):
    def __init__(self, latentDim):
        super().__init__()
        self.linearReluStack = nn.Sequential(
            nn.Linear(latentDim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.Linear(1000, 28*28),
            nn.ReLU()
        )

    def reshape(self, x):
        return x.view(-1, 1, 28, 28)

    def forward(self, x):
        x = self.linearReluStack(x)
        return self.reshape(x)

class AutoencoderDataset(datasets.FashionMNIST):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img

class RegularizedLatentDataset(torch.utils.data.Dataset):
    def __init__(self, imageDataset, encoder, decoder):
        self.imageDataset = imageDataset
        self.latentDataset = None
        self.encoder = encoder
        self.decoder = decoder
        self.latentDim = encoder(imageDataset.__getitem__(0).to(device)).shape
        print(f"RegularizedLatentDataset latent dimension: {latentDim}")
        self.nIters = 100

    def __len__(self):
        return self.imageDataset.__len__()
        
    def __getitem__(self, index):
        if self.latentDataset is None:
            self.buildLatent()

        img = self.imageDataset.__getitem__(index)
        latent = self.latentDataset[index]
        return img, latent
        
    def resetLatent(self):
        self.latentDataset = None

    def buildLatent(self):
        self.resetLatent()
        imageDataloader = DataLoader(self.imageDataset, batch_size=64, shuffle=False)
        for ind, batch in enumerate(imageDataloader):
            initialLatentVal = self.encoder(batch.to(device))
            optimizedLatentVal = self.findOptimalLatentEncoding(initialLatentVal.to(device), batch.to(device), self.decoder)
            if self.latentDataset is not None:
                self.latentDataset = torch.cat([self.latentDataset, optimizedLatentVal])
            else:
                self.latentDataset = optimizedLatentVal

            if ind % 100 == 0:
                current = ind * len(batch)
                print(f"[{current:>5d}/{len(imageDataloader.dataset):>5d}]")

    def findOptimalLatentEncoding(self, initialGuess, targetOutput, decoder):
        latentVal = initialGuess.clone().detach()
        latentVal.requires_grad_(True)
        optimizer = torch.optim.Adam([latentVal], lr=1e-3)
        lossFn = nn.MSELoss()

        for i in range(self.nIters):
            pred = decoder(latentVal)
            loss = lossFn(pred, targetOutput)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return latentVal

def trainLoop(dataloader, model, loss_fn, optimizer, batchSize):
    size = len(dataloader.dataset)
    for batch, X in enumerate(dataloader):
        # Compute prediction and loss
        if type(X) in [list, tuple]:
            X = [item.to(device) for item in X]
        else:
            X = X.to(device)
        pred = model(X)
        loss = loss_fn(pred, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * batchSize
            print(f"loss: {loss:>7f}  [{int(current / size * 100):>3d}%]")


def testLoop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred.to(device), X.to(device)).item()

    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>8f} \n")
    return test_loss


def plotSamples(dataloader, model, nSamples):
    fig, ax = plt.subplots(nSamples, 3, figsize=(6, 10))

    for i in range(nSamples):
        sample = next(iter(dataloader))
        img = sample[i].squeeze().to(device)
        pred = autoencoder(img.view(1, 1, 28, 28)).squeeze().detach()
        img = img.to("cpu")
        pred = pred.to("cpu")
        ax[i, 0].imshow(img, cmap='copper', vmin=0, vmax=1.0)
        ax[i, 1].imshow(pred, cmap='copper', vmin=0, vmax=1.0)
        ax[i, 2].imshow(torch.abs(img - pred), cmap='copper', vmin=0, vmax=1.0)
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
        ax[i, 2].axis("off")

    ax[0, 0].set_title("Input")
    ax[0, 1].set_title("Prediction")
    ax[0, 2].set_title("Diff")
    fig.savefig("results.png", dpi=300)
    

def trainModel(trainDataloader, testDataloader, autoencoder, lossFn, optimizer, batchSize, nEpochs):
    losses = []
    for epoch in range(nEpochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        trainLoop(trainDataloader, autoencoder, lossFn, optimizer, batchSize)
        loss = testLoop(testDataloader, autoencoder, lossFn)
        losses.append(loss)
    return losses
        
def applySparsification(model, sparsityLevel):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=sparsityLevel)


def cementSparsification(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, name="weight")
            except:
                pass
                

def sparsifyAutoencoderModel(autoencoder, finalSparsity, nSparsifyIterations, trainDataloader,
                             testDataloader, lossFn, batchSize, nEpochs):
    sparsityPerIteration = 1 - (1 - finalSparsity) ** (1 / nSparsifyIterations)
    losses = [[],[]]

    for sparseIter in range(nSparsifyIterations):
        print(f"Starting sparsification iteration {sparseIter + 1} / {nSparsifyIterations}")
        applySparsification(autoencoder.encoder, sparsityPerIteration)
        applySparsification(autoencoder.decoder, sparsityPerIteration)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learningRate)
        loss = trainModel(trainDataloader, testDataloader, autoencoder, lossFn, optimizer, batchSize, nEpochs)
        losses[0].append(1 - (1 - sparsityPerIteration) ** (sparseIter + 1))
        losses[1].append(loss[-1])

    cementSparsification(autoencoder.encoder)
    cementSparsification(autoencoder.decoder)
    return losses
        
if __name__ == "__main__":
    # hyperparameters
    batchSize = 512
    learningRate = 1e-4
    nEpochs = 10
    
    # model building
    latentDim = 64
    encoder = Encoder(latentDim).to(device)
    decoder = Decoder(latentDim).to(device)
    autoencoder = Autoencoder(encoder, decoder).to(device)
    
    # dataset loading
    trainingData = AutoencoderDataset(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    print(type(trainingData))
    testingData = AutoencoderDataset(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    trainDataloader = DataLoader(trainingData, batch_size=batchSize)
    testDataloader = DataLoader(testingData, batch_size=batchSize)

    lossFn = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learningRate)

    trainModel(trainDataloader, testDataloader, autoencoder, lossFn, optimizer, batchSize, nEpochs)
    
    print("Done!")

    #torch.save(encoder.state_dict(), "models/encoder")
    #torch.save(decoder.state_dict(), "models/decoder")

    plotSamples(testDataloader, autoencoder, 6)

    finalSparsity = 0.98
    nIterations = 10
    losses = sparsifyAutoencoderModel(autoencoder, finalSparsity, nIterations,
                                      trainDataloader, testDataloader, lossFn, batchSize, nEpochs)

    saveSparseModel(encoder.linearReluStack, "models/sparse_encoder")
    saveSparseModel(decoder.linearReluStack, "models/sparse_decoder")
    
    fig, ax = plt.subplots()
    ax.plot(losses[0], losses[1], losses)
    ax.set_xlabel("sparsity")
    ax.set_ylabel("tesdt loss")
    fig.savefig("sparsity_loss.png")
    
    plotSamples(testDataloader, autoencoder, 6)
    
    # # try training just the encoder with optimal latent dimension
    # autoencoderWithLatent = AutoencoderWithLatent(encoder, decoder).to(device)
    # regularizationTrainDataset = RegularizedLatentDataset(
    #     trainingData,
    #     encoder,
    #     decoder
    # )
    # regularizationTestDataset = RegularizedLatentDataset(
    #     testingData,
    #     encoder,
    #     decoder
    # )
    # trainDataloaderReg = DataLoader(regularizationTrainDataset, batch_size=batchSize)
    # testDataloaderReg = DataLoader(regularizationTestDataset, batch_size=batchSize)
    # encoderOptimizer = torch.optim.Adam(encoder.parameters(), lr=learningRate)
    # mseLossFn = nn.MSELoss()
    # lossFn = lambda yPred, yTrue: mseLossFn(yPred[0], yTrue[0]) + mseLossFn(yPred[1], yTrue[1])
    
    # for epoch in range(nEpochs*2):
    #     print(f"Epoch {epoch+1}\n-------------------------------")
    #     train_loop(trainDataloaderReg, autoencoderWithLatent, lossFn, optimizer, batchSize)
    #     #test_loop(testDataloaderReg, autoencoderWithLatent, lossFn)
    # print("Done!")

    # torch.save(encoder.state_dict(), "models/opt_encoder")
    # torch.save(decoder.state_dict(), "models/opt_decoder")
    
    # plotSamples(trainDataloader, autoencoder, 6)
    
    plt.show()
