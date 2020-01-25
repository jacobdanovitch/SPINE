import torch
from torch import nn

from modules import Encoder, Decoder

import logging
logging.basicConfig(level=logging.INFO)


class SPINEModel(torch.nn.Module):

	def __init__(self, params):
		super(SPINEModel, self).__init__()
		
		# params
		self.inp_dim = params['inp_dim']
		self.hdim = params['hdim']
		self.noise_level = params['noise_level']
		self.getReconstructionLoss = nn.MSELoss()
		self.rho_star = 1.0 - params['sparsity']
		
		# autoencoder
		logging.info("Building model ")
		self.linear1 = nn.Linear(self.inp_dim, self.hdim)
		self.linear2 = nn.Linear(self.hdim, self.inp_dim)
		

	def forward(self, batch_x, batch_y):
		
		# forward
		batch_size = batch_x.size(0)
		linear1_out = self.linear1(batch_x)
		h = linear1_out.clamp(min=0, max=1) # capped relu
		out = self.linear2(h)

		# different terms of the loss
		reconstruction_loss = self.getReconstructionLoss(out, batch_y) # reconstruction loss
		psl_loss = self._getPSLLoss(h, batch_size) 		# partial sparsity loss
		asl_loss = self._getASLLoss(h)    	# average sparsity loss
		total_loss = reconstruction_loss + psl_loss + asl_loss
		
		return out, h, total_loss, [reconstruction_loss,psl_loss, asl_loss]


	def _getPSLLoss(self,h, batch_size):
		return torch.sum(h*(1-h))/ (batch_size * self.hdim)


	def _getASLLoss(self, h):
		temp = torch.mean(h, dim=0) - self.rho_star
		temp = temp.clamp(min=0)
		return torch.sum(temp * temp) / self.hdim






class SeqSPINEModel(SPINEModel):
    def __init__(self, params):
        super(SeqSPINEModel, self).__init__(params)
        
        self.encoder = Encoder(params['seq_len'], params['hdim'], params['inp_dim'])
        self.decoder = Decoder(params['seq_len'], params['inp_dim'], params['hdim'])

    def forward(self, x):
        hidden = self.encoder(x)
        reconstructed = self.decoder(hidden)
        # different terms of the loss
        reconstruction_loss = self.getReconstructionLoss(
            reconstructed, x)  # reconstruction loss
        psl_loss = self._getPSLLoss(hidden, hidden.size(0)) 		# partial sparsity loss
        asl_loss = self._getASLLoss(hidden)    	# average sparsity loss
        total_loss = reconstruction_loss + psl_loss + asl_loss

        return reconstructed, hidden, total_loss, [reconstruction_loss, psl_loss, asl_loss]