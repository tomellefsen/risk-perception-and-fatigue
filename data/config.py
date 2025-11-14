SLICES = [
    {
        'name': 'P1_Épidémie_initiale_et_confinement',
        'start': '2020-03-11',
        'end': '2020-08-31',
        'events': [
            ('2020-03-13', 'Urgence sanitaire', 'red'), 
            ('2020-03-23', '1er confinement', 'darkred'), 
            ('2020-04-07', '10 000 cas totales', 'orange'),
            ('2020-05-06', 'Début assouplissement mesures', 'green'),
            ('2020-05-29', '50 000 cas totales', 'orange'),
            ('2020-06-08', '5000 décès', 'red'), 
            ('2020-06-18', 'Masque obligatoire en publique', 'blue'),
            ('2020-08-23', 'Début 2ème vague', 'darkred'),
        ]
    },
    {
        'name': 'P2_Deuxième_vage_et_coufre-feu',
        'start': '2020-09-01',
        'end': '2021-02-28',
        'events': [
            ('2020-09-29', '75 000 cas', 'orange'), ('2020-10-01', 'Confinement partiel', 'orange'),
            ('2020-10-24', '100 000 cas', 'orange'), ('2020-12-05', '150 000 cas', 'orange'),
            ('2020-12-14', 'Début de la vaccination', 'blue'), ('2020-12-25', 'Fermeture des commerces', 'red'),
            ('2020-12-29', '200 000 cas', 'orange'), ('2021-01-09', '1er couvre-feu (8 PM - 5 AM)', 'purple'),
            ('2021-01-21', '250 000 cas', 'orange'), ('2021-02-06', '10 000 décès', 'red'),
            ('2021-02-09', 'Détection du variant Alpha (Afrique du Sud)', 'black'),
        ]
    },
    {
        'name': 'P3_Troisième_vague_et_variant_alpha',
        'start': '2021-03-01',
        'end': '2021-07-17',
        'events': [
            ('2021-03-17', '300 000 cas', 'orange'), ('2021-03-23', '12.5% du Québec vacciné', 'green'),
            ('2021-03-29', 'Début 3ème vague (Alpha)', 'darkred'), ('2021-04-19', '25% du Québec vacciné', 'brown'),
            ('2021-04-09', 'Fermeture de la frontière QC-ON', 'brown'), ('2021-04-19', 'Détection du variant Delta (Inde)', 'black'),
            ('2021-05-17', '50% du Québec vacciné', 'green'), ('2021-05-28', 'Déconfinement', 'green'),
        ]
    },
    {
        'name': 'P4_Fourth_Wave_Delta_and_Vaccine_Passport',
        'start': '2021-07-18',
        'end': '2021-11-30',
        'events': [
            ('2021-07-18', 'Début 4ème vague (Delta)', 'darkred'), ('2021-08-24', 'Rentré scolaire', 'green'),
            ('2021-09-01', 'Passeport vaccinal', 'blue'), ('2021-09-14', '400 000 cas', 'orange'),
            ('2021-09-30', '75% du Québec vacciné (début doses rappel)', 'green'), ('2021-11-11', 'Détection du variant Omicron (Botswana)', 'black'),
            
        ]
    },
    {
        'name': 'P5_Fifth_Wave_Omicron_Explosion',
        'start': '2021-12-01',
        'end': '2022-03-31',
        'events': [
            ('2021-12-05', 'Début de la 5ème vague', 'darkred'), ('2021-12-21', 'Retour du confinement', 'red'),
            ('2021-12-22', '500 000 cas', 'orange'), ('2021-12-31', '2nd couvre-feu (10 PM - 5 AM)', 'purple'),
            ('2022-01-07', '750 000 cas', 'orange'), ('2022-01-31', 'Assouplissement des mesures', 'green'),
            ('2022-03-12', 'Début de la 6ème vague', 'darkred'),
        ]
    },
    {
        'name': 'P6_Endemic_Phase_Omicron_Subvariants',
        'start': '2022-04-01',
        'end': '2024-01-15', # End of the provided dataset
        'events': [
            ('2022-04-10', '1 000 000 cas', 'orange'), ('2022-04-21', '15 000 décès', 'orange'),
            ('2022-06-01', 'Fin urgence sanitaire', 'darkgreen'),
            ('2022-07-01', 'Début 7ème vague (BA.4/BA.5)', 'darkred'), ('2023-01-01', 'Début 8ème vague', 'darkred'),
        ]
    }
]