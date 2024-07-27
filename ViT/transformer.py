
 # divide the parts of transformers into blocks of different py files aswell !

import numpy as np
import copy 
import math 
import torch 
import torch.nn as nn
from torch.nn import functional as F



class AttentionLayer(nn.Module):

    def __init__(self,  embed_dim , dropout =0.1 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim , embed_dim , bias =False) # the weights are initialised by pytorch
        self.key_proj = nn.Linear(embed_dim, embed_dim,bias = False)
        self.value_proj = nn.Linear(embed_dim ,embed_dim, bias = False)
       # self.full_op_proj = nn.Linear(embed_dim ,embed_dim,bias = False)

       

    def forward(self, key , query , value , attn_mask = None):
        N , S , D = query.shape
        N , T , D = value.shape 

        assert value.shape == query.shape

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        dot_product = torch.einsum("NSD , NTD -> NST " , [query , key]) # score for each word 

        if attn_mask is not None:
            

            additive_mask = attn_mask.masked_fill(attn_mask==0 , float ('-inf')) # it restricts the sucessive words to communicate with future references 
            dot_product += additive_mask

            attention = torch.softmax( dot_product/(self.embed_dim ** (-0.5)) , dim =-1)  
          

            y = torch.einsum( " NST , NTD -> NSD" ,[attention , value]) # this is the final value of each word 
 
            return y # the attention y for each word is generated .
        
# class MultiHeadAttentionLayer(AttentionLayer): # 
#     def __init__(self , embed_dim , num_heads , dropout = 0.1): # 0.1 is the probability for dropout of each token 

#         super().__init__(embed_dim , dropout)
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         assert self.head_dim * num_heads == embed_dim, "embed dim is not correct"

#         self.query_proj = nn.Linear(embed_dim , embed_dim , bias =False) # the weights are initialised by pytorch
#         self.key_proj = nn.Linear(embed_dim, embed_dim,bias = False)
#         self.value_proj = nn.Linear(embed_dim ,embed_dim, bias = False)
#         self.full_op_proj = nn.Linear(embed_dim ,embed_dim,bias = False)
#         self.dropout = nn.Dropout(dropout)

#         #self.head_proj = nn.ModuleList(AttentionLayer(embed_dim , num_heads) for _ in range (num_heads)  )  # parallel processinng of heads for num_heads no. of heads 

#     def forward(self,query, key , value , attn_mask =None): 
#         H = self.num_heads
#         N , S ,D = query.shape
#         N , T,D = value.shape
#         B = self.head_dim
#         assert key.shape ==value.shape
#         assert B == D //H

#         query = self.query_proj(query)
#         key = self.key_proj(key)
#         value = self.value_proj(value)

#         query = query.view(N,S,H,B).transpose(1,2) # numheads is the mebdding dim and headdim times all of it 
#         key = key.view(N,T,H,B).transpose(1,2)
#         value = value.view(N,T,H,B).transpose(1,2)




#         dot_product = torch.einsum("NSHB ,NTHB-> NHST ",[query ,key.transpose(-2,-1)]) / np.sqrt(B)
#         if attn_mask is not None:

#             additive_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        
#             additive_mask = additive_mask.masked_fill(attn_mask==0 , float ('-inf')) 
#             dot_product += additive_mask 



#             y = self.dropout(F.softmax (dot_product,dim= -1))
#             y = torch.matmul(y , value) # NHST , NTDd/h -> NHSd/h

#             output = y.transpose(1,2).reshape(N,S,D) # concatennating into 3d tensor of NSD 

#             output = self.full_op_proj(y)
#             assert output != None

#             print(output)
#             return output 

class MultiHeadAttentionLayer(AttentionLayer):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__(embed_dim, dropout)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed dim is not correct"

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.full_op_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        H = self.num_heads
        N, S, D = query.shape
        N, T, D = value.shape
        B = self.head_dim



        assert key.shape == value.shape
        assert B == D // H

        assert D == H*B

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = query.reshape(N, S, H, B).transpose(1, 2)
        key = key.reshape(N, T, H, B).transpose(1, 2)
        value = value.reshape(N, T, H, B).transpose(1, 2)

        assert key.shape == (N,H,T,B)

        assert query.shape == (N,H,S,B)
       # assert key.shape == (N,T,H,B)
        assert value.shape == (N,H,T,B)


        dot_product = torch.einsum("NHSB, NHTB -> NHST", query, key) / math.sqrt(B)

        assert dot_product.shape == (N,H,S,T)

        if attn_mask is not None:
            additive_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            additive_mask = additive_mask.masked_fill(attn_mask == 0, float('-inf'))
            dot_product += additive_mask
            assert dot_product.shape == (N,H,S,T)



        attn_weights = F.softmax(dot_product, dim=-1)
        assert attn_weights.shape == (N,H,S,T)
        attn_weights = self.dropout(attn_weights)
        assert attn_weights.shape == (N,H,S,T)
        value = value.view(N , T,H , B)
        #attn_weights = attn_weights.view(N,H,S,T)
        assert D == H*B 
        assert attn_weights.shape == ( N,H,S,T)
        value = value.reshape(N,H,T,B)
            
             
        assert value.shape == (N,H,T,B)
       

        y = torch.matmul(attn_weights, value)

        y = y.transpose(1, 2).reshape(N, S, D)

        output = self.full_op_proj(y)
        
        # Ensure output is not None
        assert output is not None, "Output is None"
        
        print(output)
        return output

        
class PositionalEncoding(nn.Module):
###---------------
    def __init__(self , embed_dim , dropout=0.2 , max_len = 5000, device ='cuda'):
        super().__init__()

        self.embed_dim = embed_dim 
        self.encoding = torch.nn.Embedding( max_len , embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device 

        pos = torch.arange(max_len).unsqueeze(1).float()
        div_term =torch.exp(torch.arange(0,embed_dim ,2).float()*-(np.log(10000.0)/embed_dim))
        pos_encoding = torch.zeros(max_len , embed_dim)

        pos_encoding[: , 0::2] = torch.sin(pos*div_term)
        pos_encoding[: , 1::2] = torch.cos(pos*div_term)
        self.encoding.weight =nn.Parameter(pos_encoding , requires_grad=False)
    def forward ( self , x):

        N , S ,D= x.shape
       

        print(x.shape)
       
        

        #assert D == self.embed_dim , f" Expected embedding dimension { self.embed_dim}"


        # positions = torch.arange( 0 , S).unsqueeze(0).expand(N,S).to(self.device)
        # positions = positions.unsqueeze(2)
        # positions = positions.expand(N , S , D )

        #  # positions of the captions are not 
 
        # encoding = torch.sin(positions / (10000 ** (2 * torch.arange(self.embed_dim).float() /self.embed_dim)))

        # #encoding = encoding.reshape(x.shape[0] , x.shape[1]  , 256)
        # print((encoding.shape))

        pos = torch.arange(S).unsqueeze(0)
        pos_encodings = self.encoding(pos)

        output = x + pos_encodings
        output = self.dropout(output)

        return output # to be used in encoder block 
    
class SelfAttentionBlock(nn.Module):

    def __init__(self , input_dim , num_heads , dropout=0.1):
        super().__init__()
        self.self_attn = AttentionLayer(input_dim , num_heads)
        self.dropout =nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self , seq  , mask ): 
        self_attention = self.self_attn(seq ,seq,seq, mask)
        self_attention =torch.add(self_attention , seq )# residual connection to input seq

        out = self.dropout(self.layernorm(self_attention))

        return out


    


class CrossAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttentionLayer(input_dim, num_heads)
        if self.cross_attn is None:
            raise ValueError("cross_attn is None after initialization")
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, seq, cond):  
        if seq is None:
            raise ValueError("seq is None")
        if cond is None:
            raise ValueError("cond is None")

        print(f'seq: {seq}')
        print(f'cond: {cond}')

        cross_attention = self.cross_attn(seq, cond, cond)  # since key and value have same dims
        if cross_attention is None:
            raise ValueError("cross_attention is None after MultiHeadAttentionLayer")
        
        print(f'cross_attention: {cross_attention}')

        cross_attention = self.dropout(cross_attention)
        out = seq + cross_attention
        out = self.norm(out)

        return out

    

class FeedForwardBlock(nn.Module):
        
        def __init__( self , input_dim , dim_feedforward=2048 , dropout=0.1):

            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim , dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward,input_dim )
            )

            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(input_dim)



        def forward(self,seq ):
            mlp = self.mlp(seq)
            out = self.dropout(mlp)
            out = torch.add(mlp , out )

            out = self.norm( out)
##
            return out 
        

class DecoderLayer(nn.Module):
    def  __init__( self , input_dim , num_heads , dim_feedforward =2048 , dropout=0.1):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim , num_heads )
        self.cross_atn_block = CrossAttentionBlock(input_dim , num_heads )
        self.feedforward_block = FeedForwardBlock(input_dim , num_heads)


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
        self.apply(self._init_weights)

        #self.apply(weights_initialised)
        self.device = device 
        self.to(device)

    def get_data_embeddings( self , features , captions ):
            
            
            N = features.shape[0]
            D = features.shape[1]
            T = captions.shape[1]

            
            captions_embedding = self.caption_embedding(captions)
            catpions_embedding = self.positional_encoding(captions_embedding )





            feature_embedding = self.feature_embedding(features)
            feature_embedding = feature_embedding.unsqueeze(1)

            return feature_embedding, captions_embedding


    def get_causal_mask(self , _len):

            # multiplicative 
           
            mask = torch.tril(torch.ones(_len , _len)).bool()
            mask = mask.to(dtype=torch.float)
            mask = mask.masked_fill(mask == 0 , float ( '-inf'))
            

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
                module.weight.data.normal_(mean=0.0 , std = 0.02)
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
                            





               
               

            




            


        
            







        




        









                             
        











        
    

        



        




        






        

        

