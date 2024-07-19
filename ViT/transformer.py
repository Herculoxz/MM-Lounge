
 # divide the parts of transformers into blocks of different py files aswell !

import numpy as np
import copy 
import math 
import torch 
import torch.nn as nn
from torch.nn import functional as F


num_heads = 4 
class AttentionLayer(nn.Module):

    def __init__(self,  embed_dim , num_heads ): # num_heads is added by me 
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads 
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(self.embed_dim, self.embed_dim) # the weights are initialised by pytorch
        self.key = nn.Linear(self.embed_dim, self.embed_dim  , bias = False)
        self.value = nn.Linear(self.embed_dim, self.embed_dim , bias = False)
        self.full_op = nn.Linear( self.embed_dim , embed_dim)

       

    def forward(self, key , query , value , attn_mask = None):
        N , S , D = query.shape
        N , T , D = value.shape 

        assert value.shape == query.shape

        query = query.view( N,S,self.num_heads ,self.head_dim)
        key = key.view( N ,-1,self.num_heads, self.head_dim)
        value = value.view( N , -1 , self.num_heads , self.head_dim)

        dot_product = torch.einsum("Q * K = ndsk " , [query , key]) # score for each word 

        if attn_mask is not None:
            

            additive_mask = dot_product.masked_fill(attn_mask==0 , float ('-infi')) # it restricts the sucessive words to communicate with future references 
            dot_product += additive_mask

            attention = torch.softmax( dot_product/self.embed_dim ** (0.5) , dim =3)  

            y = torch.einsum( " ndst * ntdh = nsdh" ,[attention , value]).reshape( N , S ,D*self.head_dim) # this is the final value of each word 
            y = self.full_op(y) # multiplied with weighted sum
 
            return y # the attention y for each word is generated .
        
class MultiHeadAttentionLayer(AttentionLayer): # 
    def __init__(self , embed_dim , num_heads , dropout = 0.1): # 0.1 is the probability for dropout of each token 

        super().__init__(embed_dim , dropout)
        self.num_heads = num_heads

        self.head_proj = nn.ModuleList(AttentionLayer(embed_dim , num_heads) for _ in range (num_heads)  )  # parallel processinng of heads for num_heads no. of heads 

    def forward(self,query, key , value ,x, attn_mask =None): # x is the original input 
        H = self.num_heads
        N , S ,D = query.shape
        N , S , T = value.shape
        assert key.shape ==value.shape

        query = self.query()
        key = self.key()
        value = self.value()


        dot_product = torch.einsum("nsd * ntd =  ndst",[query ,key])
        if attn_mask is not None:
        
            additive_mask = dot_product.masked_fill(attn_mask==0 , float ('-infi')) 
            dot_product += additive_mask 



            y = nn.Dropout(F.softmax (dot_product))

            output = torch.cat([h(query , key , value) for h in self.head_proj] , dim =-1 )
            output = nn.Dropout(output)
            return output 
        
class PositionalEncoding(nn.Module):

    def __init__(self , embed_dim , dropout=0.2 , max_len = 5000, device ='cuda'):
        super().__init__()

        self.embed_dim = embed_dim 
        self.encoding = torch.nn.Embedding( max_len , embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device 


    def forward ( self , x):

        N , S= x.shape
        D = self.embed_dim 

        print(x.shape)
       
        

        #assert D == self.embed_dim , f" Expected embedding dimension { self.embed_dim}"


        # positions = torch.arange( 0 , S).unsqueeze(0).expand(N,S).to(self.device)
        # positions = positions.unsqueeze(2)
        # positions = positions.expand(N , S , D )

        #  # positions of the captions are not 
 
        # encoding = torch.sin(positions / (10000 ** (2 * torch.arange(self.embed_dim).float() /self.embed_dim)))

        # #encoding = encoding.reshape(x.shape[0] , x.shape[1]  , 256)
        # print((encoding.shape))

       

        output = x + self.encoding(x)
        output = self.dropout(output)

        return output # to be used in encoder block 
    
class SelfAttentionBlock(nn.Module):

    def __init__(self , input_dim , num_heads , dropout=0.1):
        super().__init__()
        self.self_attn = AttentionLayer(input_dim , num_heads)
        self.dropout =nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self , seq  , mask ): 
        self_attention = self.self_attn(seq , mask)

        out = self.dropout(self.layernorm(self_attention))

        return out


    


class CrossAttentionBlock(nn.Module):

    def __init__(self , input_dim , num_heads , dropout =0.1):

        super().__init__()

        self.cross_attn = MultiHeadAttentionLayer(input_dim , num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self , seq , cond , x): # x is the original input 

        cross_attention = self.cross_attn(seq , cond)
        out = self.dropout(self.norm(cross_attention + x ))

        return out 
    

class FeedForwardBlock(nn.Module):
        
        def __init__( self , input_dim , num_heads , dim_feedforward=2048 , dropout=0.1):

            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim , dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward,input_dim )
            )

            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(input_dim)



        def forward(self,seq ,x): # x is the original input 
            mlp = self.mlp(seq)
            out = self.dropout(mlp)

            out = self.norm( mlp + x)
##
            return out 
        

class DecoderLayer(nn.Module):
    def  __init__( self , input_dim , num_heads , dim_feedforward =2048 , dropout=0.1):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim , num_heads , dropout )
        self.cross_atn_block = CrossAttentionBlock(input_dim , num_heads , dropout)
        self.feedforward_block = FeedForwardBlock(input_dim , num_heads , dim_feedforward , dropout )


    def forward(self , seq , cond , mask):
        out = self.self_atn_block(seq , mask)
        out =self.cross_atn_block(out , cond)
        return self.feedforward_block(out)
    

class TransformerDecoder(nn.Module):

    def __init__( self , word_to_idx , idx_to_word , input_dim , embed_dim , num_heads=4 , num_layers=2 , max_length=50 , device ='cpu'):


        super().__init__()

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start=word_to_idx.get("<START" , None)
        self.idx_to_word = idx_to_word


        self.layers = nn.ModuleList([DecoderLayer(embed_dim , num_heads) for _ in range(num_layers)])
        self.caption_embedding = nn.Embedding(vocab_size ,  embed_dim  , padding_idx = self._null)
        self.positional_encoding = PositionalEncoding(embed_dim , max_len=max_length)
        self.feature_embedding = nn.Linear(input_dim , embed_dim )
        self.score_projection= nn.Linear(embed_dim , vocab_size)

        #weights_initialised = _init_weights()

        #self.apply(weights_initialised)
        self.device = device 
        self.to(device)

    def get_data_embeddings( self , features , captions ):
            
            
            N = features.shape[0]
            D = features.shape[1]
            T = captions.shape[1]

            catpions_embed = self.positional_encoding(captions )
            captions_embedding = self.caption_embedding(captions)





            feature_embedding = self.feature_embedding(features)
            feature_embedding = feature_embedding.unsqueeze(1)

            return feature_embedding, captions_embedding

            



            

    def get_causal_mask(self , _len):

            # multiplicative 
            self.register_buffer('tril', torch.tril(torch.ones(_len , _len)))
            mask = torch.tril(torch.ones(_len , _len))
            mask = mask.masked_fill(self.tril[:_len, :_len] == 0 , float ( '-inf'))

            return mask 
        

    def forward(self , features , captions):

            features_embed , captions_embed = self.get_data_embeddings(features , captions )
            mask = self.get_causal_mask(captions_embed.shape[1])
            mask.to(captions_embed.dtype)


            output = captions_embed
            for layer in self.layers:
                output = layer(output , features_embed , mask = mask)

                scores = self.score_projection(output)

                return scores
            

    def _init_weights(self , module):
            if isinstance(module , (nn.Linear , nn.Embedding)):
                module.weight.data.normal__(mean=0.0 , std = 0.02)
                if isinstance(module , nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
                elif isinstance(module , nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)


                    def sample ( self , features , max_length = 30):
                        with torch.no_grad():
                            features = torch.Tensor(features).to(self.device)
                            N = features.shape[0]

                            captions = self._null*np.ones( N , dtype=np.int32)

                            partial_caption = self._start*np.ones( N , dtypre =np.int32)
                            partial_caption= torch.LongTensor(partial_caption).to(self.device)

                            partial_caption = partial_caption.unsqueeze(1)

                            for t in range(max_length):
                                output_logits = self.forward(features , partial_caption)
                                output_logits = output_logits[:,-1,:]

                                word = torch.argmax( output_logits , axis = 1)

                                captions[:,t] =  word.cpu().numpy()

                                word = word.unsqueeze(1)

                                partial_caption = torch.cat([partial_caption , word ], dim =1 )

                                return captions 
                            





               
               

            




            


        
            







        




        









                             
        











        
    

        



        




        






        

        
