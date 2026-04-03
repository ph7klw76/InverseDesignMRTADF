# Generating Descriptor for QM9

## Problem
mordred.Calculator.pandas() on all molecules at once and padelpy.padeldescriptor() on a single giant SDF file. This fails because: PaDEL's Java JVM runs out of heap memory on large batches and silently freezes; Mordred infinite-loops on certain polycyclic topologies with no timeout mechanism; padelpy blocks the entire Python process so a single hung molecule kills everything; and writing one CSV at the end means any crash — even after 8 hours of work — loses all computed data.

## Solution
The CheckpointDB class uses SQLite in WAL (Write-Ahead Logging) journal mode, which guarantees that a crash mid-write cannot corrupt the database. Each molecule gets one row with three independent status flags (mordred_done, rdkit_done, padel_done) and descriptor values stored as zlib-compressed JSON blobs. After computing descriptors for each molecule, the result is committed atomically — so even a power failure preserves every molecule completed before that instant. On restart, the pipeline queries WHERE mordred_done = 0 to find only unprocessed molecules and skips everything already computed.
The TimeoutExecutor wraps every computation in a ProcessPoolExecutor(max_workers=1) with a hard timeout. When Mordred hangs on a ring topology or PaDEL's Java process freezes, the subprocess is terminated after the deadline (120s for Mordred, 45s for PaDEL). This is the only reliable way to kill a hung Java JVM from Python. The molecule is marked as failed (done = -1) with an error message, and the pipeline moves to the next one.
PaDEL is processed in micro-batches of 25 molecules (not 134,000 in one SDF). If a batch fails, each molecule in that batch is retried individually via padelpy.from_smiles() with its own timeout. This two-level fallback means a single toxic molecule cannot bring down the entire batch.

<img width="610" height="660" alt="image" src="https://github.com/user-attachments/assets/4efc28e3-4d49-4536-9809-65b429022e23" />

## How to use
Java is also required for PaDEL — download from https://adoptium.net/ (Temurin JDK 17 or 21), check "Add to PATH" during install, then verify with java -version in a fresh Anaconda Prompt.
How to use in Spyder:

Open robust_descriptor_pipeline.py, scroll to the bottom, edit the four paths, set max_molecules=100 to test, and press F5. If it works, set max_molecules=None for all QM9 molecules and press F5 again. If it crashes at any point — power failure, Java OOM, Mordred hang, accidental window close — just press F5 again. The pipeline reopens the checkpoint database, reports how many molecules were completed in each stage, and resumes from exactly where it stopped.

## Resource estimates for full QM9 (134k molecules):
The estimated total time is roughly 37 hours with 4 PaDEL threads, the SQLite checkpoint database will be approximately 314 MB, and the final CSV will be around 3 GB. You can monitor progress via the live progress bar with throughput and ETA, and a descriptor_pipeline.log file captures every failure with full details. The pipeline also runs gc.collect() every 200 molecules to prevent memory leaks during long runs.
