#!/bin/bash

# === CONFIGURATION ===
LOCAL_DIR=/home/gpoloudenny/Projects/CrystaLLM/out
REMOTE_USER=uvv78gt
REMOTE_HOST=jean-zay.idris.fr
REMOTE_DIR=/lustre/fswork/projects/rech/nxk/uvv78gt/CrystaLLM/out
SSH_KEY="$HOME/.ssh/id_rsa"

# === FONCTION DE SORTIE PROPRE ===
cleanup() {
  echo -e "\n🛑 [sync] Arrêt demandé. Fin du script."
  exit 0
}
trap cleanup SIGINT

# === TRANSFERT AVEC SUPPRESSION ===
echo "🚀 [sync] Début du transfert des fichiers..."

# On utilise rsync avec --remove-source-files pour supprimer les fichiers une fois transférés
rsync -avz --remove-source-files --progress -e "ssh -i $SSH_KEY" \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"

echo "✅ [sync] Transfert et nettoyage terminés."
