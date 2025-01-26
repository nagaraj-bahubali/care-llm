from datetime import date
import config as cfg
from contextlib import asynccontextmanager
from utils.async_redis_saver import AsyncRedisSaver


def update_user_profile(user_profile: dict) -> dict:
    """
    Update user profile by calculating age from birthday and removing birthday field.

    Args:
        user_profile (dict): User profile dictionary containing 'birthday' key with datetime object

    Returns:
        dict: Updated user profile with 'age' calculated and 'birthday' field removed
    """
    def calc_age(bday): return date.today().year - bday.year - \
        ((date.today().month, date.today().day) < (bday.month, bday.day))

    user_profile['age'] = calc_age(user_profile['birthday'])
    del user_profile['birthday']

    return user_profile


@asynccontextmanager
async def get_redis_checkpointer():
    """Creates an async context manager for the Redis checkpoint saver."""
    async with AsyncRedisSaver.from_conn_info(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT, db=cfg.REDIS_DB) as checkpointer:
        yield checkpointer


async def check_key_pattern_exists(key: str) -> bool:
    """
    Check if any keys matching the pattern exist in Redis using SCAN.
    Continues scanning until finding a match or completing the entire scan.
    """
    try:
        async with get_redis_checkpointer() as redis_client:
            pattern = f"checkpoint:{key}::*"
            cursor = 0

            while True:
                # Get next batch with cursor
                cursor, keys = await redis_client.conn.scan(cursor=cursor, match=pattern, count=100)

                # Return True if matching key found
                if len(keys) > 0:
                    return True

                # If cursor is 0, then scanning all keys is completed
                if cursor == 0:
                    break

            # If we get here, we've scanned everything and found no matches
            return False

    except Exception as e:
        print(f"Error checking Redis key pattern: {e}")
        return False
