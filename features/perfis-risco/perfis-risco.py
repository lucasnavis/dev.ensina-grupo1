classes_ativos = [
    "ações",
    "crypto",
    "renda_fixa",
    "fx",
    "moeda"
]

perfis_risco = {
    "conservador": [
        0.15, # Ações
        0.00, # Crypto
        0.65, # Renda fixa
        0.10, # Foreign Exchange (G5)
        0.10  # Moeda Nacional (BRL, nesse caso)
    ],
    "moderado": [
        0.40, 0.05, 0.40, 0.10, 0.05
    ],
    "agressivo": [
        0.65, 0.15, 0.10, 0.05, 0.05
    ]
}

for nome, peso in perfis_risco.items():
    assert abs(sum(peso) - 1.0) < 1e-6, f"{nome} não soma até 1"