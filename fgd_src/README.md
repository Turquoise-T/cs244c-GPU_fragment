# Fragmentation Gradient Descent

Using the *Beware of Fragmentation* paper.

### Structure

```
fgd_src/
├── scheduler.py          # Main FGD scheduler
├── worker.py            # Worker node management
├── job.py               # Job definitions
├── lease.py             # Lease management
├── fragmentation.py     # Fragmentation calculation
├── node.py              # Node representation
├── workload.py          # Workload management
├── rpc/
│   ├── scheduler_server.py
│   ├── scheduler_client.py
│   ├── worker_server.py
│   └── worker_client.py
├── docs/
│   └── atc23-weng.pdf
└── utils.py
```