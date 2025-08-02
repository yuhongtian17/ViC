_base_ = [
    '../../_base_/_hep2seq_models_/hepv2-selfsup-reg_trans-base_nofpn.py',
    '../../_base_/_hep2seq_datasets_/abla_pretrain/hep2seq-fullana-top50x.py',
    '../../_base_/_hep2seq_schedules_/schedule_12e_selfsup_trans.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        out_eng=True,
        out_phithe=False,
    ),
    head=dict(
        backbone_out_eng=True,
        backbone_out_phithe=False,
        recover_eng=False,
        recover_phithe=True,
    ))
