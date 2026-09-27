python ./symbolic_logic.py
cp ./arc-agi_* ./kaggle/combined/
python -m dataset.build_arc_dataset   --input-file-prefix kaggle/combined/arc-agi   --output-dir data/arcSymconcept-aug-100   --subsets trainingSym evaluationSym conceptSym   --test-set-name evaluationSym
#run_name="pretrain_mlp_t_sym" torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  pretrain.py arch=trm data_paths="[data/arcSymconcept-aug-100]" evaluators="[{"name":"arc@ARC"}]" epochs=2000 eval_interval=200 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.mlp_t=True arch.pos_encodings=none arch.L_layers=1 arch.H_cycles=3 arch.L_cycles=6 +run_name=${run_name} ema=True
