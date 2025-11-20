import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc

import scipy
import scvi
import sklearn
import grae

from grae.models import GRAE

import Models as mod

import torch
import torch_geometric


def split_train_val(X, y, train_size=0.85, test_size=0.15, train_batch_size=128, val_batch_size=64):
	
	X=scipy.sparse.csr_matrix(X, dtype="float32").toarray()		 
	y=np.array(y).astype(int)

	X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42, stratify=y)
	y_train=y_train.astype(int)	
	y_val=y_val.astype(int)	

	class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
	weight = 1. / class_sample_count
	samples_weight = np.array([weight[t] for t in y_train])
	samples_weight = torch.from_numpy(samples_weight)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
	
	trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))	
	train_dataloader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=train_batch_size, shuffle=False, num_workers=1, sampler = sampler)
	 

	class_sample_count = np.array([len(np.where(y_val==t)[0]) for t in np.unique(y_val)])
	weight = 1. / class_sample_count
	samples_weight = np.array([weight[t] for t in y_val])
	samples_weight = torch.from_numpy(samples_weight)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

	valDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
	val_dataloader = torch.utils.data.DataLoader(dataset = valDataset, batch_size=val_batch_size, shuffle=False, num_workers=1, sampler = sampler)

	return train_dataloader, val_dataloader

def split_train_val_test(X, y, train_size=0.7, val_size=0.1, test_size=0.2, train_batch_size=128, valtest_batch_size=64):
	X=scipy.sparse.csr_matrix(X, dtype="float32").toarray()		 
	y=np.array(y).astype(int)

	X_tv, X_test, y_tv, y_test = sklearn.model_selection.train_test_split(X, y, train_size=train_size+val_size, random_state=42, stratify=y)
	y_tv=y_tv.astype(int)
	y_test=y_test.astype(int)

	class_sample_count = np.array([len(np.where(y_test==t)[0]) for t in np.unique(y_test)])
	weight = 1. / class_sample_count
	samples_weight = np.array([weight[t] for t in y_test])
	samples_weight = torch.from_numpy(samples_weight)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
	
	testDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
	test_dataloader = torch.utils.data.DataLoader(dataset = testDataset, batch_size=valtest_batch_size, shuffle=False, num_workers=1, sampler = sampler)
	
	
	X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_tv, y_tv, train_size=train_size/(train_size + val_size), random_state=42, stratify=y_tv)
	y_train=y_train.astype(int)
	y_val=y_val.astype(int)

	class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
	weight = 1. / class_sample_count
	samples_weight = np.array([weight[t] for t in y_train])
	samples_weight = torch.from_numpy(samples_weight)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
	
	trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
	train_dataloader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=train_batch_size, shuffle=False, num_workers=1, sampler = sampler)
	 

	class_sample_count = np.array([len(np.where(y_val==t)[0]) for t in np.unique(y_val)])
	weight = 1. / class_sample_count
	samples_weight = np.array([weight[t] for t in y_val])
	samples_weight = torch.from_numpy(samples_weight)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

	valDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
	val_dataloader = torch.utils.data.DataLoader(dataset = valDataset, batch_size=valtest_batch_size, shuffle=False, num_workers=1, sampler = sampler)

	return train_dataloader, val_dataloader, test_dataloader


	
# ++++++++++++++++++++++++++++++ TRAIN GNNs

def GNN_1_step_training(model, train_loader, optimizer, criterion, GNN):
	if not (GNN == "GCN" or GNN == "GAT"):
		raise ValueError(f'Parameter GNN cannot be {GNN}')
		
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_loss=0
	train_f1w=0
	for batch in train_loader:
		optimizer.zero_grad()
		batch = batch.to(device)
		if GNN == "GCN":
			out = model(batch.x, batch.edge_index, batch.edge_weight)[:batch.batch_size]
		elif GNN == "GAT":
			out = model(batch.x, batch.edge_index)[:batch.batch_size]
		else:
			raise ValueError(f'Parameter GNN cannot be {GNN}')
		
		# NOTE Only consider predictions and labels of seed nodes:
		y = batch.y[:batch.batch_size]
		loss_batch = criterion(out, y)
		loss_batch.backward()
		optimizer.step()
		train_loss+=loss_batch.item()
		train_f1w+=sklearn.metrics.precision_recall_fscore_support(y.detach().numpy(), out.argmax(dim=1).detach().numpy(), average="weighted")[2]
	
	return train_loss/len(train_loader), train_f1w/len(train_loader)

def GNN_validation(model, val_loader, optimizer, criterion, GNN):
	if not (GNN == "GCN" or GNN == "GAT"):
		raise ValueError(f'Parameter GNN cannot be {GNN}')
		
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	val_f1w=0
	val_loss=0
	with torch.no_grad(): 
		for batch in val_loader:
			batch = batch.to(device)
			if GNN == "GCN":
				out = model(batch.x, batch.edge_index, batch.edge_weight)[:batch.batch_size]
			elif GNN == "GAT":
				out = model(batch.x, batch.edge_index)[:batch.batch_size]
			else:
				raise ValueError(f'Parameter GNN cannot be {GNN}')
			y = batch.y[:batch.batch_size]
			loss_batch = criterion(out, y)
			val_loss += loss_batch.item()
			val_f1w += sklearn.metrics.precision_recall_fscore_support(out.argmax(dim=1).detach().numpy(), y.detach().numpy(), average="weighted")[2]
	
	return val_loss/len(val_loader), val_f1w/len(val_loader)
	
def GNN_train_node_classifier(model, data, optimizer, criterion, model_name, GNN, epochs=300, patience=30):
	if not (GNN == "GCN" or GNN == "GAT"):
		raise ValueError(f'Parameter GNN cannot be {GNN}')

	best_val_f1w = -1
	best_epoch = -1
	history={}
	history["TrainLoss"]=[]
	history["TrainF1W"]=[]
	history["ValLoss"]=[]
	history["ValF1W"]=[]
	   
	train_loader = torch_geometric.loader.NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[3,2], batch_size=128, directed=False, shuffle=True)
	val_loader = torch_geometric.loader.NeighborLoader(data, input_nodes=data.val_mask, num_neighbors=[3,2], batch_size=64, directed=False, shuffle=True)
	
	for epoch in range(0, epochs):
		### Training
		train_loss, train_f1w = GNN_1_step_training(model, train_loader, optimizer, criterion, GNN)
		history["TrainLoss"].append(np.around(train_loss, decimals=5))
		history["TrainF1W"].append(np.around(train_f1w, decimals=5))

		### Validation
		val_loss, val_f1w = GNN_validation(model, val_loader, optimizer, criterion, GNN)
		history["ValLoss"].append(np.around(val_loss, decimals=5))
		history["ValF1W"].append(np.around(val_f1w, decimals=5))

		### Early stopping
		if val_f1w > best_val_f1w:
			best_val_f1w = val_f1w
			best_epoch = epoch
			torch.save(model.state_dict(), model_name)
			print(f"{GNN} checkpoint! Best epoch {best_epoch} | Best val loss {val_loss:.3f} | Best val F1W {best_val_f1w:.3f}", flush=True)
		elif epoch - best_epoch > patience:
			model.load_state_dict(torch.load(model_name))
			val_loss, best_val_f1w = GNN_validation(model, val_loader, optimizer, criterion, GNN)
			print(f"{GNN} early stopped at epoch {epoch} with best epoch {best_epoch} | Best val loss: {val_loss:.3f} | Best val F1W: {best_val_f1w:.3f}", flush=True)
			return model, history  
	
	model.load_state_dict(torch.load(model_name))
	return model, history

# ++++++++++++++++++++++++++++++ TRAIN NN

def NN_1_step_training(model, train_loader, optimizer, criterion):  
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_loss=0
	train_f1w=0
	for X, y in train_loader:
		optimizer.zero_grad()
		X = X.to(device)
		out = model(X)
		
		loss_batch = criterion(out, y)
		loss_batch.backward()
		optimizer.step()
		train_loss+=loss_batch.item()
		train_f1w+=sklearn.metrics.precision_recall_fscore_support(y.detach().numpy(), out.argmax(dim=1).detach().numpy(), average="weighted")[2]
	return train_loss/len(train_loader), train_f1w/len(train_loader)

def NN_validation(model, val_loader, optimizer, criterion):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	val_f1w=0
	val_loss=0
	with torch.no_grad(): 
		for X, y in val_loader:
			optimizer.zero_grad()
			X = X.to(device)
			out = model(X)
			
			loss_batch = criterion(out, y)
			val_loss += loss_batch.item()
			val_f1w += sklearn.metrics.precision_recall_fscore_support(y.detach().numpy(), out.argmax(dim=1).detach().numpy(), average="weighted")[2]
	
	return val_loss/len(val_loader), val_f1w/len(val_loader)

def NN_train_classifier(model, adata, optimizer, criterion, model_name, epochs=300, patience=30):
	best_val_f1w = -1
	best_epoch = -1
	history={}
	history["TrainLoss"]=[]
	history["TrainF1W"]=[]
	history["ValLoss"]=[]
	history["ValF1W"]=[]

	train_loader, val_loader, test_laoder=split_train_val_test(adata.X, np.array(adata.obs.target))
	
	for epoch in range(0, epochs):
		### Training
		train_loss, train_f1w = NN_1_step_training(model, train_loader, optimizer, criterion)
		history["TrainLoss"].append(np.around(train_loss, decimals=5))
		history["TrainF1W"].append(np.around(train_f1w, decimals=5))

		### Validation
		val_loss, val_f1w = NN_validation(model, val_loader, optimizer, criterion)
		history["ValLoss"].append(np.around(val_loss, decimals=5))
		history["ValF1W"].append(np.around(val_f1w, decimals=5))

		### Early stopping
		if val_f1w > best_val_f1w:
			best_val_f1w = val_f1w
			best_epoch = epoch
			torch.save(model.state_dict(), model_name)
			print(f"NN checkpoint! Best epoch {best_epoch} | Best val loss {val_loss:.3f} | Best val F1W {best_val_f1w:.3f}", flush=True)
		elif epoch - best_epoch > patience:
			model.load_state_dict(torch.load(model_name))
			val_loss, best_val_f1w = NN_validation(model, val_loader, optimizer, criterion)
			print(f"NN early stopped at epoch {epoch} with best epoch {best_epoch} | Best val loss: {val_loss:.3f} | Best val F1W: {best_val_f1w:.3f}", flush=True)
			return model, history, train_loader, val_loader, test_laoder
	
	model.load_state_dict(torch.load(model_name))
	return model, history, train_loader, val_loader, test_laoder


####### Apply AE
def PeakVI(M, patience=30, epochs=300, model_name="PippoPeakVI"):
	to_use=sc.AnnData(M)
	scvi.model.PEAKVI.setup_anndata(adata=to_use)
	model = scvi.model.PEAKVI(adata=to_use)
	model.train(max_epochs=epochs, early_stopping_patience=patience)
	model.save(model_name, overwrite=True)
	return model.get_latent_representation(), scipy.sparse.csr_matrix(model.get_accessibility_estimates(), dtype="float32")
	
def scVI(M, patience=30, epochs=300, model_name="PipposcVI"):
	to_use=sc.AnnData(M)
	scvi.model.LinearSCVI.setup_anndata(to_use)
	model = scvi.model.LinearSCVI(to_use, n_hidden=int(to_use.shape[1]**(1/2)), n_latent=int(to_use.shape[1]**(1/3)))
	model.train(max_epochs=epochs, early_stopping=True, check_val_every_n_epoch=1, early_stopping_min_delta=0.01, early_stopping_patience=patience)
	model.save(model_name, overwrite=True)
	return model.get_latent_representation(), scipy.sparse.csr_matrix(model.get_normalized_expression(n_sample=10), dtype="float32")

def GR_AE(M, y=[], epochs=300, model_name="PippoGRAE"):
	y=np.array(y).astype(int)
	M=scipy.sparse.csr_matrix(M, dtype="float32").todense()

	dataset = BaseDataset(M, y=y, split='none', split_ratio=1, random_state=42, labels=y)
	train_dataset, val_dataset, val_mask = dataset.validation_split(ratio=0.15)

	model = GRAE(epochs=epochs, patience=30, latent_dim=int(np.round(M.shape[1] ** (1/3))), write_path=model_name, data_val=val_dataset)

	model.fit(train_dataset)

	return model.transform(dataset), scipy.sparse.csr_matrix(model.inverse_transform(model.transform(dataset)), dtype="float32")


def TAE(M, y=[], params=None, model_name="PippoTAE"):

	if params is None:
		params={}
		params["hidden_dim"]=int(M.shape[1]**(1/2))
		params["latent_dim"]=int(M.shape[1]**(1/3))
		params["dp"]=0.4
		params["lam"]=1
		params["p"]=2
		params["lr"]=2e-04
		params["weight_decay"]=4e-03
		params["epochs"]=300
		params["patience"]=30

	print("TAE params:", params, flush=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	if len(y) == 0:
		y=np.ones(shape=(M.shape[0],))
	y=np.array(y).astype(int)
	train_dataloader, val_dataloader = split_train_val(M, y)

	ae_kwargs={key : params[key] for key in ["hidden_dim","latent_dim","dp"]}
	ae_kwargs["input_dim"]=M.shape[1]
	Topo=mod.TopologicallyRegularizedAutoencoder(ae_kwargs=ae_kwargs, lam=params["lam"], p=params["p"])

	optimizer = torch.optim.Adam(Topo.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
	
	best_val_loss = 10e10
	best_epoch = -1
	torch.save(Topo.state_dict(), f"{model_name}.pth")

	for epoch in range(0, params["epochs"]):
		val_losses=[]
		for X, Y in train_dataloader:
			X=torch.tensor(scipy.sparse.csr_matrix(X, dtype="float32").todense()).to(device)
			train_loss = Topo(X)
			optimizer.zero_grad()
			train_loss.backward()
			optimizer.step()

		for X, Y in val_dataloader:
			X=torch.tensor(scipy.sparse.csr_matrix(X, dtype="float32").todense()).to(device)
			val_losses.append(Topo(X).item())
		val_loss=np.mean(val_losses)
		
		if np.around(val_loss/best_val_loss, decimals=2) < 0.95:
			best_val_loss = val_loss
			best_epoch = epoch
			print(f"TAE checkpoint! Best epoch {best_epoch} | Best loss {best_val_loss:.7f}", flush=True)
			torch.save(Topo.state_dict(), f"{model_name}.pth")
		
		elif epoch - best_epoch > params["patience"]:
			Topo.load_state_dict(torch.load(f"{model_name}.pth"))
			Topo.eval()
			val_losses=[]
			for X, Y in val_dataloader:
				X=torch.tensor(scipy.sparse.csr_matrix(X, dtype="float32").todense()).to(device)
				val_losses.append(Topo(X).item())
			best_val_loss=np.mean(val_losses)
			print(f"TAE early stopped at epoch {epoch} with best epoch {best_epoch} | Best loss: {best_val_loss:.7f}", flush=True)
			dataset = torch.tensor(scipy.sparse.csr_matrix(M, dtype="float32").todense())
			return Topo.encode(dataset).detach().numpy(), scipy.sparse.csr_matrix(Topo.decode(Topo.encode(dataset)).detach().numpy(),  dtype="float32")

	print(f"TAE stopped at epoch {epoch} with best epoch {best_epoch}", flush=True)
	Topo.load_state_dict(torch.load(f"{model_name}.pth"))
	Topo.eval()
	dataset = torch.tensor(scipy.sparse.csr_matrix(M, dtype="float32").todense())
	return Topo.encode(dataset).detach().numpy(), scipy.sparse.csr_matrix(Topo.decode(Topo.encode(dataset)).detach().numpy(),  dtype="float32")
	

def AE(M, y=[], params=None, model_name="PippoAE"):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if params is None:
		params={}
		params["hidden_dim"]=int(M.shape[1]**(1/2))
		params["latent_dim"]=int(M.shape[1]**(1/3))
		params["dp"]=0.4
		params["lr"]=7e-04
		params["weight_decay"]=2e-03
		params["epochs"]=300
		params["patience"]=30

	print("AE params:", params, flush=True)
	
	if len(y) == 0:
		y=np.ones(shape=(M.shape[0],))
	y=np.array(y).astype(int)
	train_dataloader, val_dataloader = split_train_val(M, y)
					
	ae = mod.BaseAE(input_dim=M.shape[1], hidden_dim=params["hidden_dim"], latent_dim=params["latent_dim"], dp=params["dp"])

	optimizer = torch.optim.Adam(ae.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
	
	best_val_loss = 10e10
	best_epoch = -1
	loss_fn=torch.nn.MSELoss()

	torch.save(ae.state_dict(), f"{model_name}.pth")

	for epoch in range(0, params["epochs"]):
		val_losses=[]
		for X,y in train_dataloader:
			X_hat = ae(X)
			reconstruction_loss = loss_fn(X, X_hat)
			optimizer.zero_grad()
			reconstruction_loss.backward()
			optimizer.step()

		for X, y in val_dataloader:
			X_hat = ae(X)
			reconstruction_loss = loss_fn(X, X_hat)
			val_losses.append(reconstruction_loss.item())
		val_loss=np.mean(val_losses)

		if np.around(val_loss/best_val_loss, decimals=2) < 0.95:
			best_val_loss = val_loss
			best_epoch = epoch
			print(f"AE checkpoint! Best epoch {best_epoch} | Best loss {best_val_loss:.7f}", flush=True)
			torch.save(ae.state_dict(), f"{model_name}.pth")
		elif epoch - best_epoch > params["patience"]:
			ae.load_state_dict(torch.load(f"{model_name}.pth"))
			ae.eval()
			val_losses=[]
			for X, y in val_dataloader:
				X_hat = ae(X)
				reconstruction_loss = loss_fn(X, X_hat)
				val_losses.append(reconstruction_loss.item())
			best_val_loss=np.mean(val_losses)
			print(f"AE early stopped at epoch {epoch} with best epoch {best_epoch} | Best loss: {best_val_loss:.7f}", flush=True)
			dataset = torch.tensor(scipy.sparse.csr_matrix(M, dtype="float32").todense())
			return ae.encode(dataset).detach().numpy(), scipy.sparse.csr_matrix(ae.decode(ae.encode(dataset)).detach().numpy(),  dtype="float32") 

	print(f"AE stopped at epoch {epoch} with best epoch {best_epoch}", flush=True)
	ae.load_state_dict(torch.load(f"{model_name}.pth"))
	ae.eval()
	dataset = torch.tensor(scipy.sparse.csr_matrix(M, dtype="float32").todense())
	return ae.encode(dataset).detach().numpy(), scipy.sparse.csr_matrix(ve.decode(ae.encode(dataset)).detach().numpy(),  dtype="float32") 


def VAE(M, y=[], params=None, model_name="PippoVAE"):

	if params is None:
		params={}
		params["hidden_dim"]=int(M.shape[1]**(1/2))
		params["latent_dim"]=int(M.shape[1]**(1/3))
		params["dp"]=0.3
		params["lr"]=1e-02
		params["weight_decay"]=1e-02
		params["epochs"]=300
		params["patience"]=30

	print("VAE params:", params, flush=True)
	
	if len(y) == 0:
		y=np.ones(shape=(M.shape[0],))
	y=np.array(y).astype(int)
	train_dataloader, val_dataloader = split_train_val(M, y)
					
	ae_kwargs={key : params[key] for key in ["hidden_dim","latent_dim","dp"]}
	ae_kwargs["input_dim"]=M.shape[1]
	vae = mod.VAutoencoder(ae_kwargs=ae_kwargs)

	optimizer = torch.optim.Adam(vae.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
	
	best_val_loss = 10e10
	best_epoch = -1
	loss_fn=torch.nn.MSELoss()

	annealer = mod.KLAnnealer(
        mode=params.get("anneal_mode","linear"),
        n_steps=params.get("anneal_steps",10000),
        beta_start=0.0,
        beta_max=0.5)
	global_step = 0

	torch.save(vae.state_dict(), f"{model_name}.pth")

	for epoch in range(0, params["epochs"]):
		val_losses=[]
		for X,y in train_dataloader:
			global_step += 1
			beta = annealer.step()
			X_hat, mu, log_var = vae(X)
			reconstruction_loss = loss_fn(X, X_hat)
			kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim = 1), dim = 0)
			train_loss = reconstruction_loss + beta * kl_loss
			optimizer.zero_grad()
			train_loss.backward()
			optimizer.step()

		for X, y in val_dataloader:
			X_hat, mu, log_var = vae(X)
			reconstruction_loss = loss_fn(X, X_hat)
			kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim = 1), dim = 0)
			beta = annealer.beta(global_step)   # use current beta
			tot_loss = reconstruction_loss + beta * kl_loss
			val_losses.append(tot_loss.item())
		val_loss=np.mean(val_losses)

		if np.around(val_loss/best_val_loss, decimals=2) < 0.95:
			best_val_loss = val_loss
			best_epoch = epoch
			print(f"VAE checkpoint! Best epoch {best_epoch} | Best loss {best_val_loss:.7f}", flush=True)
			torch.save(vae.state_dict(), f"{model_name}.pth")
		elif epoch - best_epoch > params["patience"]:
			vae.load_state_dict(torch.load(f"{model_name}.pth"))
			vae.eval()
			val_losses=[]
			for X, y in val_dataloader:
				X_hat, mu, log_var = vae(X)
				reconstruction_loss = loss_fn(X, X_hat)
				kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim = 1), dim = 0)
				beta = annealer.beta(global_step)   # use current beta
				tot_loss = reconstruction_loss + beta * kl_loss
				val_losses.append(tot_loss.item())
			best_val_loss=np.mean(val_losses)
			print(f"VAE early stopped at epoch {epoch} with best epoch {best_epoch} | Best loss: {best_val_loss:.7f}", flush=True)
			dataset = torch.tensor(scipy.sparse.csr_matrix(M, dtype="float32").todense())
			return vae.encode(dataset)[0].detach().numpy(), scipy.sparse.csr_matrix(vae.decode(vae.encode(dataset)[0]).detach().numpy(),  dtype="float32") 

	print(f"VAE stopped at epoch {epoch} with best epoch {best_epoch}", flush=True)
	vae.load_state_dict(torch.load(f"{model_name}.pth"))
	vae.eval()
	dataset = torch.tensor(scipy.sparse.csr_matrix(M, dtype="float32").todense())
	return vae.encode(dataset)[0].detach().numpy(), scipy.sparse.csr_matrix(vae.decode(vae.encode(dataset)[0]).detach().numpy(),  dtype="float32") 



def embbedding_and_graph(adata, y=None, representation=None, layer="X", model_name="Pappo", params=None):
	
	if layer == "X":
		M=adata.X.copy()
	else:
		M=adata.layers[layer].copy()
	
	if representation == "GRAE":
		Z = GR_AE(M, y=y)[0]
		ad_ret=sc.AnnData(scipy.sparse.csr_matrix(Z, dtype="float32"))
		del Z
		sc.pp.neighbors(ad_ret, use_rep="X", method="umap")
		return scipy.sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32"), scipy.sparse.csr_matrix(ad_ret.X, dtype="float32")
	
	elif representation == "AE":
		Z = AE(M, y=y, model_name=model_name, params=params)[0]
		ad_ret=sc.AnnData(scipy.sparse.csr_matrix(Z, dtype="float32"))
		del Z
		sc.pp.neighbors(ad_ret, use_rep="X", method="umap")
		return scipy.sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32"), scipy.sparse.csr_matrix(ad_ret.X, dtype="float32")

	elif representation == "VAE":
		Z = VAE(M, y=y, model_name=model_name, params=params)[0]
		ad_ret=sc.AnnData(scipy.sparse.csr_matrix(Z, dtype="float32"))
		del Z
		sc.pp.neighbors(ad_ret, use_rep="X", method="umap")
		return scipy.sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32"), scipy.sparse.csr_matrix(ad_ret.X, dtype="float32")


	elif representation == "TAE":
		Z = TAE(M, y=y, model_name=model_name, params=params)[0]
		ad_ret=sc.AnnData(scipy.sparse.csr_matrix(Z, dtype="float32"))
		sc.pp.neighbors(ad_ret, use_rep="X", method="umap")
		return scipy.sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32"), scipy.sparse.csr_matrix(ad_ret.X, dtype="float32")


	elif representation == "PeakVI":
		Z = PeakVI(M)[0]
		ad_ret=sc.AnnData(scipy.sparse.csr_matrix(Z, dtype="float32"))
		del Z
		sc.pp.neighbors(ad_ret, use_rep="X", method="umap")
		return scipy.sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32"), scipy.sparse.csr_matrix(ad_ret.X, dtype="float32")


	elif representation == "scVI":
		Z = scVI(M)[0]
		ad_ret=sc.AnnData(scipy.sparse.csr_matrix(Z, dtype="float32"))
		del Z
		sc.pp.neighbors(ad_ret, use_rep="X", method="umap")
		return scipy.sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32"), scipy.sparse.csr_matrix(ad_ret.X, dtype="float32")

	else:
		raise ValueError(f'Parameter representation cannot be {representation}')

def ApplyAE(M, representation, y=None, model_name="Pippo"):
	
	if representation == "GRAE":
		return GR_AE(M, y=y, model_name=model_name)

	if representation == "AE":
		return AE(M, y=y, model_name=model_name)

	elif representation == "TAE":
		return TAE(M, y=y, model_name=model_name)

	elif representation == "VAE":
		return VAE(M, y=y, model_name=model_name)

	elif representation == "PeakVI":
		return PeakVI(M, model_name=model_name)
	
	elif representation == "scVI":
		return scVI(M, model_name=model_name)

	else:
		raise ValueError(f'Parameter representation cannot be {representation}')
