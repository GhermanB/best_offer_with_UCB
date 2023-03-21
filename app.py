"""
Small service on FastAPI for choosing the best offer based on reward we got with feedback
"""
import uvicorn
from fastapi import FastAPI
import numpy as np

app = FastAPI()

offers = {}
clicks_offers = {}


@app.on_event('startup')
def startup_event():
    offers.clear()
    clicks_offers.clear()


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """Sample offer wuth UCB"""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    # Create new offers if not existed
    for i in offers_ids:
        if i not in offers:
            offers[i] = {
                "offer_id": i,
                "clicks": 0,
                "conversions": 0,
                "reward": 0,
                "cr": 0,
                "rpc": 0
            }

    total = np.array([])
    conv = np.array([])
    reward = np.array([])

    for i in offers_ids:
        total = np.append(total, offers[i]['clicks'])
        reward = np.append(reward, offers[i]['reward'])
        conv = np.append(conv, offers[i]['conversions'])

    # UCB
    rew_squared = reward ** 2
    log_total = np.log(total)
    results = np.divide(rew_squared, conv, out=np.zeros_like(rew_squared), where=conv != 0) \
              - np.divide(reward, conv, out=np.zeros_like(reward), where=conv != 0) ** 2 \
              + np.sqrt(np.divide(log_total, conv, out=np.zeros_like(log_total), where=conv != 0))

    final_idx = np.argmax(results)
    offer_id = offers_ids[final_idx]

    offers[offer_id]['clicks'] += 1
    clicks_offers[click_id] = offer_id

    # Prepare response
    response = {
        "click_id": click_id,
        "offer_id": offer_id
    }

    return response


@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """Get feedback for particular click"""
    # Response body consists of click ID
    # and accepted click status (True/False)

    offer_id = clicks_offers[click_id]

    if reward > 0:
        offers[offer_id]['reward'] += reward
        offers[offer_id]['conversions'] += 1
        is_conversion = True
    else:
        is_conversion = False

    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": is_conversion,
        "reward": reward
    }
    return response


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """Return offer's statistics"""

    if offer_id not in offers:
        offers[offer_id] = {
            "offer_id": offer_id,
            "clicks": 0,
            "conversions": 0,
            "reward": 0,
            "cr": 0,
            "rpc": 0,
        }

    if offers[offer_id]['clicks'] != 0:
        rpc = offers[offer_id]['reward'] / offers[offer_id]['clicks']
    else:
        rpc = 0
    if offers[offer_id]['conversions'] != 0:
        cr = offers[offer_id]['conversions'] / offers[offer_id]['clicks']
    else:
        cr = 0

    response = {
        "offer_id": offer_id,
        "clicks": offers[offer_id]['clicks'],
        "conversions": offers[offer_id]['conversions'],
        "reward": offers[offer_id]['reward'],
        "cr": cr,
        "rpc": rpc,
    }

    return response


def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="localhost")


if __name__ == "__main__":
    main()
