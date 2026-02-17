import numpy as np
import pandas as pd
import scanpy as sc
import seagall as sgl
import os, time, threading, gc, psutil, sklearn

# ---------- Helper: peak RSS (incl. child processes) while a callable runs ----------
def _rss_now(proc: psutil.Process, include_children: bool = True) -> int:
	"""Return current RSS in bytes for proc (+ children if include_children)."""
	rss = 0
	try:
		rss = proc.memory_info().rss
	except psutil.Error:
		pass
	if include_children:
		for c in proc.children(recursive=True):
			try:
				rss += c.memory_info().rss
			except psutil.Error:
				pass
	return rss

def run_with_peak_increase(fn, *args, poll_s: float = 0.01, include_children: bool = True, discard_result: bool = False, **kwargs):
	"""
	Measure ONLY the function's incremental RAM footprint.
	Returns: (result_or_None, time_sec, peak_increase_bytes)

	- Baseline is sampled right before starting the thread.
	- We poll RSS and report max(RSS - baseline).
	- If discard_result=True, the function's return value is immediately dropped
	  to avoid counting persistently stored outputs as "algorithmic" usage.
	"""
	gc.collect()  # reduce noise from previous work
	proc = psutil.Process(os.getpid())

	def rss_now():
		rss = 0
		try: rss = proc.memory_info().rss
		except psutil.Error: pass
		if include_children:
			for c in proc.children(recursive=True):
				try: rss += c.memory_info().rss
				except psutil.Error: pass
		return rss

	# Baseline: after GC, before the function starts
	baseline = rss_now()
	peak_increase = 0
	done, out, err = False, None, None

	def runner():
		nonlocal out, err, done
		try:
			out = fn(*args, **kwargs)
		except BaseException as e:
			err = e
		finally:
			done = True

	t = threading.Thread(target=runner)
	t.start()
	start = time.time()
	while not done:
		inc = rss_now() - baseline
		if inc > peak_increase:
			peak_increase = inc
		time.sleep(poll_s)
	t.join()
	elapsed = time.time() - start

	# Final sample in case the last spike is right before return
	inc = rss_now() - baseline
	if inc > peak_increase:
		peak_increase = inc

	if discard_result:
		# Drop the function's return value so persistent outputs aren't counted
		out = None
		gc.collect()

	if err:
		raise err
	return out, elapsed, max(0, peak_increase)


adata=sc.read_h5ad("BigDataset.h5ad")

sizes=np.array([10, 15, 25, 35, 50, 75, 100, 175, 250, 350, 500, 1000])*1000

# Make sure output dirs exist
os.makedirs("Time", exist_ok=True)
os.makedirs("Tables", exist_ok=True)

# ---------- Benchmark loop ----------
rows = []
for n_cells in sizes:
	out=sklearn.model_selection.train_test_split(adata.obs.index, test_size=n_cells, stratify=adata.obs["author_cell_type"])[1]
	adata_sub=adata[out]
	print(f"Starting with {n_cells}", flush=True)

	# geometrical_graph — algorithmic footprint only (drop outputs)
	_, t_geom, peak_geom = run_with_peak_increase(
		sgl.ee.geometrical_graph,
		adata_sub,
		target_label="author_cell_type",
		path=f"Time/test_{n_cells}",
		discard_result=True)  # don't keep outputs; pure algorithmic footprint
	
	# explain — algorithmic footprint only
	_, t_explain, peak_explain = run_with_peak_increase(
		sgl.ee.explain,
		adata_sub,
		target_label="author_cell_type",
		path=f"Time/test_{n_cells}",
		discard_result=True)
	
	rows.append({
		"n_cells": adata_sub.n_obs,
		"time_geom_sec": t_geom,
		"time_explain_sec": t_explain,
		"peak_geom_gb": peak_geom / 1e9,
		"peak_explain_gb": peak_explain / 1e9})

	print(f"[{adata_sub.n_obs:>9} cells] "
		  f"time: geom={t_geom:.2f}s explain={t_explain:.2f}s total={t_geom+t_explain:.2f}s | "
		  f"peak: geom={peak_geom/1e9:.2f}GB explain={peak_explain/1e9:.2f}GB", flush=True)

	# Save results
	df = pd.DataFrame(rows)
	df.to_csv("Tables/Time.tsv", sep="\t", index=False)
	print("Saved: Tables/Time.tsv", flush=True)
