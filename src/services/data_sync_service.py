"""
Data Synchronization Service
"""
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.clients.msa_backend_client import ApiClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataSyncService:
    def __init__(self, api_client: ApiClient):
        self.api_client = api_client
        self.sync_status_path = Path("data/sync_status.json")
        self._ensure_data_dir_exists()

    def _ensure_data_dir_exists(self):
        self.sync_status_path.parent.mkdir(exist_ok=True)

    def _get_last_sync_time(self) -> Optional[datetime]:
        if not self.sync_status_path.exists():
            return None
        try:
            with open(self.sync_status_path, "r") as f:
                data = json.load(f)
                return datetime.fromisoformat(data["last_sync_time"])
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return None

    def _set_last_sync_time(self, sync_time: datetime):
        with open(self.sync_status_path, "w") as f:
            json.dump({"last_sync_time": sync_time.isoformat()}, f)

    async def run_sync(self):
        logger.info("=" * 60)
        logger.info("Data synchronization process started.")
        
        last_sync_time = self._get_last_sync_time()
        if last_sync_time:
            logger.info(f"Last sync time: {last_sync_time.isoformat()}")
        else:
            logger.info("No last sync time found. Performing a full sync.")

        current_sync_time = datetime.now(timezone.utc)

        try:
            members, posts, applies, bookmarks, comments = await asyncio.gather(
                self.api_client.get_members(last_sync_time),
                self.api_client.get_recruit_posts(last_sync_time),
                self.api_client.get_apply_records(last_sync_time),
                self.api_client.get_bookmarks(last_sync_time),
                self.api_client.get_comments(last_sync_time),
            )

            logger.info(f"Fetched {len(members)} members.")
            logger.info(f"Fetched {len(posts)} recruit posts.")
            logger.info(f"Fetched {len(applies)} apply records.")
            logger.info(f"Fetched {len(bookmarks)} bookmarks.")
            logger.info(f"Fetched {len(comments)} comments.")
            
            # Here, you would typically save the data to the database.
            # For now, we are just logging the counts.
            
            self._set_last_sync_time(current_sync_time)
            logger.info(f"Successfully updated last sync time to {current_sync_time.isoformat()}")

        except Exception as e:
            logger.error(f"Data synchronization failed: {e}", exc_info=True)
        
        logger.info("Data synchronization process finished.")
        logger.info("=" * 60)
