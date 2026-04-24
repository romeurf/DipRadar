import os
import time
import logging
from universe import load_universe
from scanner import get_global_quote, crossed_drop_threshold, REQUEST_SLEEP
from state import already_alerted, mark_alerted, clear_old_state
from telegram_client import send_telegram

DROP_THRESHOLD = float(os.environ.get('DROP_THRESHOLD', '10'))
MAX_SYMBOLS_PER_RUN = int(os.environ.get('MAX_SYMBOLS_PER_RUN', '25'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def run_scan():
    universe = load_universe()
    
    if not universe:
        logging.info("Universe vazio. A criar universe_us.csv...")
        universe = build_universe()
        save_universe(universe)
    
    elif universe_is_stale(days=7):
        logging.info("Universe desatualizado. A refrescar universe_us.csv...")
        universe = build_universe()
        save_universe(universe)
    
    if not universe:
        logging.info("Não foi possível carregar/criar o universo.")
        return

    checked = 0
    for row in universe:
        if checked >= MAX_SYMBOLS_PER_RUN:
            break

        symbol = row['symbol']
        if already_alerted(symbol):
            continue

        try:
            quote = get_global_quote(symbol)
            checked += 1
            if not quote:
                time.sleep(REQUEST_SLEEP)
                continue

            if crossed_drop_threshold(quote, DROP_THRESHOLD):
                msg = (
                    f"📉 {symbol}\n"
                    f"Preço: {quote['price']}\n"
                    f"Fecho anterior: {quote['previous_close']}\n"
                    f"Queda: {quote['change_percent']:.2f}%"
                )
                send_telegram(msg)
                mark_alerted(symbol)
                logging.info(f'Alerta enviado: {symbol}')
            else:
                logging.info(f"{symbol}: {quote['change_percent']:.2f}%")

        except Exception as e:
            logging.error(f'{symbol}: {e}')

        time.sleep(REQUEST_SLEEP)

    clear_old_state()


if __name__ == '__main__':
    run_scan()
