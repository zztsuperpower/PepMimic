# default
CODE_DIR=`realpath $(dirname "$0")/..`
NAME=PeptideMimicry
AECONFIG=${CODE_DIR}/configs/train_autoencoder.yaml
PreLDMCONFIG=${CODE_DIR}/configs/pretrain_ldm.yaml


OUTLOG=./exps/$NAME/output.log

echo "Pretraining LDM with config $PreLDMCONFIG:" >> $OUTLOG
cat $PreLDMCONFIG >> $OUTLOG

AE_CKPT=${CODE_DIR}/checkpoints/ae_model.ckpt
echo "Using Autoencoder checkpoint: ${AE_CKPT}" >> $OUTLOG
if [ "$PRETRAIN_LDM_FLAG" = "1" ]; then
    bash scripts/train.sh $PreLDMCONFIG --trainer.config.save_dir=$PRE_LDM_SAVE_DIR --model.autoencoder_ckpt=$AE_CKPT
fi
