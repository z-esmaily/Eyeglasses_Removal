import torch
import torch.nn as nn
import torch.nn.functional as F

c_dim = 3
gf_dim = 64
df_dim = 64

def cnn_block(in_channels,out_channels,kernel_size,stride=1,padding=0, first_layer = False):

   if first_layer:
       return nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
   else:
       return nn.Sequential(
           nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
           nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
           )

def tcnn_block(in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0, first_layer = False):
   if first_layer:
       return nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding)

   else:
       return nn.Sequential(
           nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding),
           nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
           )

class Generator(nn.Module):
 def __init__(self,instance_norm=False):#input : 256x256
   super(Generator,self).__init__()
   self.e1 = cnn_block(c_dim,gf_dim,4,2,1, first_layer = True)
   self.e2 = cnn_block(gf_dim,gf_dim*2,4,2,1,)
   self.e3 = cnn_block(gf_dim*2,gf_dim*4,4,2,1,)
   self.e4 = cnn_block(gf_dim*4,gf_dim*8,4,2,1,)
   self.e5 = cnn_block(gf_dim*8,gf_dim*8,4,2,1,)
   self.e6 = cnn_block(gf_dim*8,gf_dim*8,4,2,1,)
   self.e7 = cnn_block(gf_dim*8,gf_dim*8,4,2,1,)
   self.e8 = cnn_block(gf_dim*8,gf_dim*8,4,2,1, first_layer=True)

   self.d1 = tcnn_block(gf_dim*8,gf_dim*8,4,2,1)
   self.d2 = tcnn_block(gf_dim*8*2,gf_dim*8,4,2,1)
   self.d3 = tcnn_block(gf_dim*8*2,gf_dim*8,4,2,1)
   self.d4 = tcnn_block(gf_dim*8*2,gf_dim*8,4,2,1)
   self.d5 = tcnn_block(gf_dim*8*2,gf_dim*4,4,2,1)
   self.d6 = tcnn_block(gf_dim*4*2,gf_dim*2,4,2,1)
   self.d7 = tcnn_block(gf_dim*2*2,gf_dim*1,4,2,1)
   self.d8 = tcnn_block(gf_dim*1*2,c_dim,4,2,1, first_layer = True) #256x256
   self.tanh = nn.Tanh()

 def forward(self,x):
   e1 = self.e1(x)
   e2 = self.e2(F.leaky_relu(e1,0.2))
   e3 = self.e3(F.leaky_relu(e2,0.2))
   e4 = self.e4(F.leaky_relu(e3,0.2))
   e5 = self.e5(F.leaky_relu(e4,0.2))
   e6 = self.e6(F.leaky_relu(e5,0.2))
   e7 = self.e7(F.leaky_relu(e6,0.2))
   e8 = self.e8(F.leaky_relu(e7,0.2))
   d1 = torch.cat([F.dropout(self.d1(F.relu(e8)),0.5,training=True),e7],1)
   d2 = torch.cat([F.dropout(self.d2(F.relu(d1)),0.5,training=True),e6],1)
   d3 = torch.cat([F.dropout(self.d3(F.relu(d2)),0.5,training=True),e5],1)
   d4 = torch.cat([self.d4(F.relu(d3)),e4],1)
   d5 = torch.cat([self.d5(F.relu(d4)),e3],1)
   d6 = torch.cat([self.d6(F.relu(d5)),e2],1)
   d7 = torch.cat([self.d7(F.relu(d6)),e1],1)
   d8 = self.d8(F.relu(d7))

   return self.tanh(d8)


class Discriminator(nn.Module):
 def __init__(self,instance_norm=False):#input : 256x256
   super(Discriminator,self).__init__()
   self.conv1 = cnn_block(c_dim*2,df_dim,4,2,1, first_layer=True) # 128x128
   self.conv2 = cnn_block(df_dim,df_dim*2,4,2,1)# 64x64
   self.conv3 = cnn_block(df_dim*2,df_dim*4,4,2,1)# 32 x 32
   self.conv4 = cnn_block(df_dim*4,df_dim*8,4,1,1)# 31 x 31
   self.conv5 = cnn_block(df_dim*8,1,4,1,1, first_layer=True)# 30 x 30

   self.sigmoid = nn.Sigmoid()
 def forward(self, x, y):
   O = torch.cat([x,y],dim=1)
   O = F.leaky_relu(self.conv1(O),0.2)
   O = F.leaky_relu(self.conv2(O),0.2)
   O = F.leaky_relu(self.conv3(O),0.2)
   O = F.leaky_relu(self.conv4(O),0.2)
   O = self.conv5(O)

   return self.sigmoid(O)
