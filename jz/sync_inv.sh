#!/bin/bash

# === CONFIGURATION ===
LOCAL_DIR=/home/gpoloudenny/Projects/CrystaLLM
REMOTE_USER=uvv78gt
REMOTE_HOST=jean-zay.idris.fr
REMOTE_DIR=/lustre/fswork/projects/rech/nxk/uvv78gt/CrystaLLM
SSH_KEY="$HOME/.ssh/id_rsa"
FICHIER_LISTE="$LOCAL_DIR/jz/.rsync_inv"

# === FONCTION DE SORTIE PROPRE ===
cleanup() {
  echo -e "\n🛑 [sync] Arrêt demandé. Fin du script."
  exit 0
}
trap cleanup SIGINT

# === SYNCHRONISATION UNIQUE ===
echo "🚀 [sync] Lancement de la synchronisation..."
echo "📄 [sync] Lecture de $FICHIER_LISTE"

while IFS= read -r chemin_relatif; do
  # Chemin absolu local
  SRC="$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/$chemin_relatif"
  echo "🔁 [sync] Transfert de : $chemin_relatif"
  PARENT_DIR="$(dirname "$chemin_relatif")"
  rsync -avz --progress -e "ssh -i $SSH_KEY" "$SRC" "$LOCAL_DIR/$PARENT_DIR/"
done < "$FICHIER_LISTE"
echo "✅ [sync] Synchronisation inverse terminée."
