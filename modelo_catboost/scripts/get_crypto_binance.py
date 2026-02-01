import ccxt
import pandas as pd
import os
import time

# 1. Configuração
exchange = ccxt.binance()
tickers = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT', 
           'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT']
caminho_csv = os.path.join(os.getcwd(), "crypto_5y_ohlcv.csv")

# 5 anos em milissegundos
five_years_ago = exchange.milliseconds() - (5 * 365 * 24 * 60 * 60 * 1000)

print("--- Iniciando Download via Binance (CCXT) ---")
all_data = []

for symbol in tickers:
    try:
        print(f"Baixando {symbol}...")
        # Baixa os dados em blocos (a Binance limita 1000 velas por vez)
        since = five_years_ago
        temp_list = []
        
        while since < exchange.milliseconds():
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since)
            if not ohlcv: break
            temp_list.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000 # Pula para o próximo dia
            time.sleep(0.1) # Evita spam

        df = pd.DataFrame(temp_list, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['Ticker'] = symbol.replace('/USDT', '')
        all_data.append(df)
        print(f"✅ {symbol} concluído! ({len(df)} dias)")

    except Exception as e:
        print(f"⚠️ Erro em {symbol}: {e}")

# 3. Salvar
if all_data:
    final_df = pd.concat(all_data)
    final_df.to_csv(caminho_csv, index=False)
    print(f"\n🚀 SUCESSO! Arquivo salvo com 5 anos de dados em: {caminho_csv}")