import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from color import *
import pygame, os, sys
from pygame.locals import *
from random import randint
pygame.init()
FPS = 30
FramePerSec = pygame.time.Clock()
GREY  = (127, 127, 127)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BACK  = (247, 247, 247)
icon = pygame.image.load('icon.png')
DISPLAYSURF = pygame.display.set_mode((500,560))
pygame.display.set_caption('color classifier')
pygame.display.set_icon(icon)
DISPLAYSURF.fill(BACK)
gameMode = 1
allLabels = [[0] * 50 for i in range(56)]
inputR,inputG,inputB = 0,0,0
mouse_x,mouse_y = 0,0
isClicking = False
lookup = {
     0: 'red',
     1: 'green',
     2: 'blue',
     3: 'orange',
     4: 'yellow',
     5: 'pink',
     6: 'purple',
     7: 'brown',
     8: 'white',
     9: 'black'
}

colors = {
    0: (255, 0, 0),
    1: (0, 128, 0),
    2: (0, 0, 255),
    3: (255, 165, 0),
    4: (255, 255, 0),
    5: (255, 192, 203),
    6: (128, 0, 128),
    7: (139, 69, 19),
    8: (255, 255, 255),
    9: (0, 0, 0)
}

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 6)
        self.fc4 = nn.Linear(6, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
net = Net()
net.load_state_dict(torch.load('model.pth'))
net.eval()
dataset = ColorDataset()
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

dataiter = iter(dataloader)
for i in range(56):
    for j in range(50):
        data = dataiter.next()
        features, labels = data
        labels = labels.numpy()
        labels = int(labels[0][0])
        allLabels[i][j] = labels

def GenerateNewColor():
    global inputR,inputG,inputB
    inputR = randint(0, 255)
    inputG = randint(0, 255)
    inputB = randint(0, 255)

def WriteToFile(num):
    OUTPUT = open("data.csv", "a+")
    line = str(inputR/255) + "," + str(inputG/255) + "," + str(inputB/255) + "," + str(num) + "\n"
    OUTPUT.write(line)
    OUTPUT.close()


def DrawScene():
    global inputR,inputG,inputB
    DISPLAYSURF.fill(BACK)
    if(gameMode == 2):
        font = pygame.font.Font("freesansbold.ttf", 32)
        text = font.render('What Is This Color?', True, BLACK, BACK)
        textRect = text.get_rect()
        textRect.center = (500//2, 20)
        DISPLAYSURF.blit(text, textRect)
        font = pygame.font.Font("freesansbold.ttf", 20)
        pygame.draw.rect(DISPLAYSURF,tuple([inputR, inputG, inputB]),(114,60,255,255))
        
        pygame.draw.rect(DISPLAYSURF,GREY,(30,348,140,40))
        pygame.draw.circle(DISPLAYSURF,(255, 0, 0),(50, 368), 14)
        text = font.render('RED', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (105, 368)
        DISPLAYSURF.blit(text, textRect)    

        pygame.draw.rect(DISPLAYSURF,GREY,(180,348,140,40))
        pygame.draw.circle(DISPLAYSURF,(0, 128, 0),(200, 368), 14)
        text = font.render('GREEN', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (265, 368)
        DISPLAYSURF.blit(text, textRect)  
        
        pygame.draw.rect(DISPLAYSURF,GREY,(330,348,140,40))
        pygame.draw.circle(DISPLAYSURF,(0, 0, 255),(350, 368), 14)
        text = font.render('BLUE', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (410, 368)
        DISPLAYSURF.blit(text, textRect)  
        
        pygame.draw.rect(DISPLAYSURF,GREY,(30,398,140,40))
        pygame.draw.circle(DISPLAYSURF,(255, 165, 0),(50, 418), 14)
        text = font.render('ORANGE', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (115, 418)
        DISPLAYSURF.blit(text, textRect)
        
        pygame.draw.rect(DISPLAYSURF,GREY,(180,398,140,40))
        pygame.draw.circle(DISPLAYSURF,(255, 255, 0),(200, 418), 14)
        text = font.render('YELLOW', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (265, 418)
        DISPLAYSURF.blit(text, textRect) 
        
        pygame.draw.rect(DISPLAYSURF,GREY,(330,398,140,40))
        pygame.draw.circle(DISPLAYSURF,(255, 192, 203),(350, 418), 14)
        text = font.render('PINK', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (410, 418)
        DISPLAYSURF.blit(text, textRect) 
        
        pygame.draw.rect(DISPLAYSURF,GREY,(30,448,140,40))
        pygame.draw.circle(DISPLAYSURF,(128, 0, 128),(50, 468), 14)
        text = font.render('PURPLE', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (115, 468)
        DISPLAYSURF.blit(text, textRect)
        
        pygame.draw.rect(DISPLAYSURF,GREY,(180,448,140,40))
        pygame.draw.circle(DISPLAYSURF,(139, 69, 19),(200, 468), 14)
        text = font.render('BROWN', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (265, 468)
        DISPLAYSURF.blit(text, textRect)
        
        pygame.draw.rect(DISPLAYSURF,GREY,(330,448,140,40))
        pygame.draw.circle(DISPLAYSURF,(255, 255, 255),(350, 468), 14)
        text = font.render('WHITE', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (410, 468)
        DISPLAYSURF.blit(text, textRect)

        font = pygame.font.Font("freesansbold.ttf", 26)
        pygame.draw.rect(DISPLAYSURF,GREY,(30,498,140,40))
        text = font.render('GO BACK', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (100, 518)
        DISPLAYSURF.blit(text, textRect)

        pygame.draw.rect(DISPLAYSURF,GREY,(180,498,140,40))
        pygame.draw.circle(DISPLAYSURF,(0, 0, 0),(200, 518), 14)
        text = font.render('BLACK', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (265, 518)
        DISPLAYSURF.blit(text, textRect)
    elif(gameMode == 1):
        font = pygame.font.Font("freesansbold.ttf", 32)
        text = font.render('Color Classifier', True, BLACK, BACK)
        textRect = text.get_rect()
        textRect.center = (500//2, 40)
        DISPLAYSURF.blit(text, textRect)

        pygame.draw.rect(DISPLAYSURF,GREY,(20,200,460,80))
        pygame.draw.rect(DISPLAYSURF,GREY,(20,300,460,80))
        pygame.draw.rect(DISPLAYSURF,GREY,(20,400,460,80))

        text = font.render('Classify', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (250, 240)
        DISPLAYSURF.blit(text, textRect)

        text = font.render('Submit Data', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (250, 340)
        DISPLAYSURF.blit(text, textRect)

        text = font.render('Visualize Dataset', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (250, 440)
        DISPLAYSURF.blit(text, textRect)
    elif(gameMode == 3):
        font = pygame.font.Font("freesansbold.ttf", 32)
        text = font.render('Color Classifier', True, BLACK, BACK)
        textRect = text.get_rect()
        textRect.center = (500//2, 20)
        DISPLAYSURF.blit(text, textRect)
        font = pygame.font.Font("freesansbold.ttf", 20)
        pygame.draw.rect(DISPLAYSURF,tuple([inputR, inputG, inputB]),(125,60,255,255))

        X = torch.tensor([inputR/255, inputG/255, inputB/255])
        pred = torch.argmax(net(X.view(-1, 3)))
        pred = pred.detach().numpy()
        pred = np.append(pred, pred)
        pred = pred[0]
        pred = lookup[pred]
        if(inputR >= 230 and inputG >= 230 and inputB >= 230):
            pred = "white"
        text = font.render(pred, True, BLACK, BACK)
        textRect = text.get_rect()
        textRect.center = (500//2, 335)
        DISPLAYSURF.blit(text, textRect)

        font = pygame.font.Font("freesansbold.ttf", 25)
        pygame.draw.rect(DISPLAYSURF,BLACK,(164,370,255,12))
        pygame.draw.rect(DISPLAYSURF,BLACK,(164,420,255,12))
        pygame.draw.rect(DISPLAYSURF,BLACK,(164,470,255,12))

        text = font.render('Red: ', True, (255, 0, 0), BACK)
        textRect = text.get_rect()
        textRect.center = (114, 375)
        DISPLAYSURF.blit(text, textRect)

        text = font.render('Green: ', True, (0, 255, 0), BACK)
        textRect = text.get_rect()
        textRect.center = (101, 425)
        DISPLAYSURF.blit(text, textRect)

        text = font.render('Blue: ', True, (0, 0, 255), BACK)
        textRect = text.get_rect()
        textRect.center = (111, 475)
        DISPLAYSURF.blit(text, textRect)

        font = pygame.font.Font("freesansbold.ttf", 26)
        pygame.draw.rect(DISPLAYSURF,GREY,(15,508,140,40))
        text = font.render('GO BACK', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (85, 528)
        DISPLAYSURF.blit(text, textRect)

        pygame.draw.circle(DISPLAYSURF,(127, 127, 127),(164+inputR, 374), 13)
        pygame.draw.circle(DISPLAYSURF,(127, 127, 127),(164+inputG, 424), 13)
        pygame.draw.circle(DISPLAYSURF,(127, 127, 127),(164+inputB, 474), 13)
    elif(gameMode == 4):

        y_offset = 0
        for i in range(56):
            x_offset = 0
            for j in range(50):
                pygame.draw.rect(DISPLAYSURF,colors[allLabels[i][j]],(x_offset,y_offset,10,10))
                x_offset += 10
            y_offset += 10
        
        font = pygame.font.Font("freesansbold.ttf", 26)
        pygame.draw.rect(DISPLAYSURF,GREY,(15,508,140,40))
        text = font.render('GO BACK', True, BLACK, GREY)
        textRect = text.get_rect()
        textRect.center = (85, 528)
        DISPLAYSURF.blit(text, textRect)
        
GenerateNewColor()

while True:
    DrawScene()
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            isClicking = True
        if event.type == pygame.MOUSEBUTTONUP:
            isClicking = False
            if(gameMode == 2):
                if(mouse_x >= 30 and mouse_x <= 30+140 and mouse_y >= 348 and mouse_y <= 348+40):
                    WriteToFile(0)
                    GenerateNewColor()
                elif(mouse_x >= 180 and mouse_x <= 180+140 and mouse_y >= 348 and mouse_y <= 348+40):
                    WriteToFile(1)
                    GenerateNewColor()
                elif(mouse_x >= 330 and mouse_x <= 330+140 and mouse_y >= 348 and mouse_y <= 348+40):
                    WriteToFile(2)
                    GenerateNewColor()
                elif(mouse_x >= 30 and mouse_x <= 30+140 and mouse_y >= 398 and mouse_y <= 398+40):
                    WriteToFile(3)
                    GenerateNewColor()
                elif(mouse_x >= 180 and mouse_x <= 180+140 and mouse_y >= 398 and mouse_y <= 398+40):
                    WriteToFile(4)
                    GenerateNewColor()
                elif(mouse_x >= 330 and mouse_x <= 330+140 and mouse_y >= 398 and mouse_y <= 398+40):
                    WriteToFile(5)
                    GenerateNewColor()
                elif(mouse_x >= 30 and mouse_x <= 30+140 and mouse_y >= 448 and mouse_y <= 448+40):
                    WriteToFile(6)
                    GenerateNewColor()
                elif(mouse_x >= 180 and mouse_x <= 180+140 and mouse_y >= 448 and mouse_y <= 448+40):
                    WriteToFile(7)
                    GenerateNewColor()
                elif(mouse_x >= 330 and mouse_x <= 330+140 and mouse_y >= 448 and mouse_y <= 448+40):
                    WriteToFile(8)
                    GenerateNewColor()
                elif(mouse_x >= 180 and mouse_x <= 180+140 and mouse_y >= 498 and mouse_y <= 498+40):
                    WriteToFile(9)
                    GenerateNewColor()
                elif(mouse_x >= 30 and mouse_x <= 30+140 and mouse_y >= 498 and mouse_y <= 498+40):
                    gameMode = 1
            elif(gameMode == 1):
                if(mouse_x >= 20 and mouse_x <= 20+460 and mouse_y >= 200 and mouse_y <= 200+80):
                    gameMode = 3
                    inputR,inputG,inputB = 0,0,0
                elif(mouse_x >= 20 and mouse_x <= 20+460 and mouse_y >= 300 and mouse_y <= 300+80):
                    gameMode = 2
                    GenerateNewColor()
                elif(mouse_x >= 20 and mouse_x <= 20+460 and mouse_y >= 400 and mouse_y <= 400+80):
                    gameMode = 4
            elif(gameMode == 3):
                if(mouse_x >= 15 and mouse_x <= 15+140 and mouse_y >= 508 and mouse_y <= 508+40):
                    inputR,inputG,inputB = 0,0,0
                    gameMode = 1
            elif(gameMode == 4):
                if(mouse_x >= 15 and mouse_x <= 15+140 and mouse_y >= 508 and mouse_y <= 508+40):
                    gameMode = 1
            
    if(isClicking and gameMode == 3):
        if(mouse_x >= 164 and mouse_x <= 164+255 and mouse_y >= 370 and mouse_y <= 370+12):
            inputR = mouse_x-164
        elif(mouse_x >= 164 and mouse_x <= 164+255 and mouse_y >= 420 and mouse_y <= 420+12):
            inputG = mouse_x-164
        elif(mouse_x >= 164 and mouse_x <= 164+255 and mouse_y >= 470 and mouse_y <= 470+12):
            inputB = mouse_x-164
    mouse_x, mouse_y = pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1]
