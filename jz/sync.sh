#!/bin/bash

# === CONFIGURATION ===
LOCAL_DIR=/home/gpoloudenny/Projects/CrystaLLM
REMOTE_USER=uvv78gt
REMOTE_HOST=jean-zay.idris.fr
REMOTE_DIR=/lustre/fswork/projects/rech/nxk/uvv78gt/CrystaLLM
SSH_KEY="$HOME/.ssh/id_rsa"
EXCLUDE_FILE_INV="$LOCAL_DIR/jz/.rsync"

# === FONCTION DE SORTIE PROPRE ===
cleanup() {
  echo -e "\n🛑 [sync] Arrêt demandé. Fin du script."
  exit 0
}

# Piéger Ctrl+C pour exécuter cleanup()
trap cleanup SIGINT

while IFS= read -r path; do
  # Skip empty lines or whitespace-only lines
  [[ -z "$path" || "$path" =~ ^[[:space:]]*$ ]] && continue

  FULL_PATH="$LOCAL_DIR/$path"

  if [ -d "$FULL_PATH" ]; then
    echo "📤 Transfert du répertoire : $path"
    rsync -azv -e "ssh -i $SSH_KEY" "$FULL_PATH/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/$path"
  elif [ -f "$FULL_PATH" ]; then
    echo "📤 Transfert du fichier : $path"
    # Crée le dossier distant s'il n'existe pas déjà
    ssh -i "$SSH_KEY" "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p \"$(dirname "${REMOTE_DIR}/$path")\""
    rsync -azv -e "ssh -i $SSH_KEY" "$FULL_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/$path"
  else
    echo "⚠️  Chemin non trouvé ou type non pris en charge : $path"
  fi
done < "$EXCLUDE_FILE_INV"
