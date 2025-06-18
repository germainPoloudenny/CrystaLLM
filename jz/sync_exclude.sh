#!/bin/bash

# === CONFIGURATION ===
LOCAL_DIR=/home/gpoloudenny/Projects/CrystaLLM
REMOTE_USER=uvv78gt
REMOTE_HOST=jean-zay.idris.fr
REMOTE_DIR=/lustre/fswork/projects/rech/nxk/uvv78gt/CrystaLLM
SSH_KEY="$HOME/.ssh/id_rsa"
EXCLUDE_FILE="$LOCAL_DIR/jz/.rsync_exclude"

# === FONCTION DE SORTIE PROPRE ===
cleanup() {
  echo -e "\n🛑 [sync] Arrêt demandé. Fin du script."
  exit 0
}
trap cleanup SIGINT

# === SYNCHRONISATION COMPLÈTE ===
echo "🚀 [sync] Lancement de la synchronisation complète..."
echo "📁 [sync] Dossier local : $LOCAL_DIR"
echo "📁 [sync] Dossier distant : $REMOTE_DIR"
echo "🚫 [sync] Fichiers exclus : $EXCLUDE_FILE"

rsync -azv \
  --exclude-from="$EXCLUDE_FILE" \
  -e "ssh -i $SSH_KEY" \
  "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

echo -e "\n✅ [sync] Synchronisation terminée."
