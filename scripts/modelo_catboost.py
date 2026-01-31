import pandas as pd
import numpy as np
import optuna
import os
import warnings
import random
import gc

from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# FUNÇÃO DE TREINO + OTIMIZAÇÃO
# ============================================================
def avaliar_modelo(X, y, split_idx, nome_modelo, h_atual):

    X_train_full, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    gap_dinamico = h_atual
    tscv = TimeSeriesSplit(n_splits=5, gap=gap_dinamico)

    def objective(trial):

        param = {
            "iterations": 1000,
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.1, log=True
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0, 1
            ),

            # ✅ ROBUSTO CONTRA CLASSES AUSENTES
            "auto_class_weights": "Balanced",

            "loss_function": "MultiClass",
            "logging_level": "Silent",
            "random_seed": SEED,
            "thread_count": -1
        }

        scores = []

        for t, v in tscv.split(X_train_full):

            # fold inválido (classe única)
            if len(np.unique(y_train_full.iloc[t])) < 2:
                continue

            model = CatBoostClassifier(**param)

            model.fit(
                X_train_full.iloc[t],
                y_train_full.iloc[t],
                eval_set=(X_train_full.iloc[v], y_train_full.iloc[v]),
                early_stopping_rounds=30,
                verbose=False
            )

            preds = model.predict(X_train_full.iloc[v])

            scores.append(
                f1_score(
                    y_train_full.iloc[v],
                    preds,
                    average="macro"
                )
            )

        # 🔒 OPÇÃO B — penalizar trials instáveis
        if len(scores) >= 2:
            return float(np.mean(scores) - np.std(scores))
        else:
            return 0.0


    print(f"\n>>> OTIMIZANDO: {nome_modelo} (50 Trials) | GAP: {gap_dinamico} <<<")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )
    study.optimize(objective, n_trials=50)

    # =========================
    # TREINO FINAL
    # =========================
    melhores_params = study.best_params.copy()

    best_model = CatBoostClassifier(
        **melhores_params,
        iterations=1000,
        auto_class_weights="Balanced",
        loss_function="MultiClass",
        logging_level="Silent",
        random_seed=SEED,
        thread_count=-1
    )

    best_model.fit(X_train_full, y_train_full)

    # =========================
    # SALVAR MODELO
    # =========================
    pasta_modelos = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models"
    )
    os.makedirs(pasta_modelos, exist_ok=True)

    nome_arquivo_modelo = os.path.join(
        pasta_modelos,
        f"catboost_{nome_modelo}.cbm"
    )

    best_model.save_model(nome_arquivo_modelo)
    print(f"✅ Modelo salvo em: {nome_arquivo_modelo}")

    # =========================
    # AVALIAÇÃO FINAL
    # =========================
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)

    loss_valor = log_loss(
        y_test,
        probs,
        labels=best_model.classes_
    )

    importancias = (
        pd.Series(
            best_model.get_feature_importance(),
            index=X.columns
        )
        .sort_values(ascending=False)
    )

    conteudo_final = (
        f"\n{'='*80}\nRESULTADOS: {nome_modelo}\n{'='*80}\n"
        f"Melhores Parâmetros: {study.best_params}\n"
        f"Acurácia: {accuracy_score(y_test, preds):.2%}\n"
        f"F1-Score Macro: {f1_score(y_test, preds, average='macro'):.4f}\n"
        f"Log Loss (Cross-Entropy): {loss_valor:.4f}\n"
        f"\nRelatório de Classificação:\n{classification_report(y_test, preds)}\n"
        f"\nMatriz de Confusão:\n{confusion_matrix(y_test, preds)}\n"
        f"\nImportância das Variáveis:\n{importancias.to_string()}\n"
    )

    print(conteudo_final)

    with open("resultados_completos_discretizacao.txt", "a") as f:
        f.write(conteudo_final)

    gc.collect()


# ============================================================
# MAIN
# ============================================================
def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_normal = os.path.join(
        script_dir, "dados_normalizados", "normal", "train"
    )
    base_fuzzy = os.path.join(
        script_dir, "dados_normalizados", "fuzzy", "train"
    )

    if not os.path.exists(base_normal):
        print("❌ ERRO CRÍTICO: Pasta de treino não encontrada!")
        return

    arquivos_normal = [
        f for f in os.listdir(base_normal)
        if f.endswith("_normal_train.csv")
    ]

    print(f" Encontrados {len(arquivos_normal)} tickers para processar.")

    horizontes = [1, 7, 30]

    for arquivo in arquivos_normal:

        ticker = arquivo.split("_")[0]

        print("\n" + "="*60)
        print(f"📊 PROCESSANDO ATIVO: {ticker}")
        print("="*60)

        try:
            df_n = pd.read_csv(os.path.join(base_normal, arquivo))

            arquivo_fuzzy = f"{ticker}_fuzzy_train.csv"
            caminho_fuzzy = os.path.join(base_fuzzy, arquivo_fuzzy)

            if not os.path.exists(caminho_fuzzy):
                print(f" Pulo: Arquivo Fuzzy {arquivo_fuzzy} não encontrado.")
                continue

            df_f = pd.read_csv(caminho_fuzzy)

            df_n["date"] = pd.to_datetime(df_n["date"])
            df_f["date"] = pd.to_datetime(df_f["date"])

            print(f" Arquivos carregados para {ticker}. Iniciando horizontes...")

            for h in horizontes:

                print(f"\n Analisando Horizonte: {h} dia(s)...")

                df_temp = df_n.copy()
                df_temp["ret_futuro"] = (
                    df_temp["close"].pct_change(h).shift(-h)
                )
                df_temp = df_temp.dropna(subset=["ret_futuro"])

                if len(df_temp) < 100:
                    print(f" Dados insuficientes para H{h}. Pulando.")
                    continue

                lim_split = int(len(df_temp) * 0.8)

                _, bins = pd.qcut(
                    df_temp["ret_futuro"].iloc[:lim_split],
                    3,
                    retbins=True,
                    labels=[0, 1, 2]
                )

                bins[0], bins[-1] = -np.inf, np.inf

                df_temp["target"] = pd.cut(
                    df_temp["ret_futuro"],
                    bins=bins,
                    labels=[0, 1, 2]
                )

                c_norm = [
                    "mom_7d", "vov_60d", "dd_dur_60d",
                    "autocorr_30d", "fgi_vol_30d"
                ]

                c_fuzz = [
                    "trend_bear", "trend_neutral", "trend_bull",
                    "vol_low", "vol_mid", "vol_high",
                    "stress_low", "stress_mid", "stress_high",
                    "ac_negative", "ac_neutral", "ac_positive",
                    "macro_calm", "macro_neutral", "macro_stress"
                ]

                cols_para_trazer = ["date", "target"] + c_norm

                df_final = (
                    pd.merge(
                        df_f,
                        df_temp[cols_para_trazer],
                        on="date",
                        how="inner"
                    )
                    .dropna()
                )

                if df_final.empty:
                    print(f" Merge vazio para {ticker} H{h}.")
                    continue

                y = df_final["target"].astype(int)
                idx_split = int(len(df_final) * 0.8)

                print(f" Otimizando Modelo NORMAL para {ticker} H{h}...")
                avaliar_modelo(
                    df_final[c_norm],
                    y,
                    idx_split,
                    f"NORMAL_{ticker}_H{h}",
                    h
                )

                print(f" Otimizando Modelo FUZZY para {ticker} H{h}...")
                avaliar_modelo(
                    df_final[c_fuzz],
                    y,
                    idx_split,
                    f"FUZZY_{ticker}_H{h}",
                    h
                )

        except Exception as e:
            print(f" Erro ao processar {ticker}: {e}")
            continue

    print("\n" + "#"*60)
    print(" PROCESSO FINALIZADO! Verifique o arquivo de resultados.")
    print("#"*60)


if __name__ == "__main__":
    main()