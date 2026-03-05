export const allTasksByPeriodByTeam: any[] = [
    // Tâches existantes
    {
        task_id: 1,
        task_title: 'Book meeting room',
        task_description: 'wxc swdfzef zef zef zef',
        duree: 3, /* duree = event.done.time - event.inProgress.time */
        executor_account_id: 1,
        category_id: 1,
        category_title: 'Category 1',
        category_description: 'Voici la Desc de la cat1'
    },
    {
        task_id: 2,
        task_title: 'Do a Explaination Report',
        task_description: 'lo lorem lorem ipsum data',
        duree: 6,
        executor_account_id: 1,
        category_id: 2,
        category_title: 'Category 2',
        category_description: 'Voici la Desc de la cat2'
    },
    {
        task_id: 3,
        task_title: 'd dz aze doi',
        task_description: 'zerz erz rzfrez  uyyk wsef',
        duree: 3,
        executor_account_id: 1,
        category_id: 2,
        category_title: 'Category 2',
        category_description: 'Voici la Desc de la cat2'
    },
    {
        task_id: 4,
        task_title: 'Book a book to book',
        task_description: 'wxc swdfzef zef zef zef',
        duree: 3,
        executor_account_id: 2,
        category_id: 2,
        category_title: 'Category 2',
        category_description: 'Voici la Desc de la cat2'
    },
    {
        task_id: 5,
        task_title: 'taratata',
        task_description: 'lo lorem lorem ipsum data',
        duree: 6,
        executor_account_id: 3,
        category_id: 3,
        category_title: 'Category 3',
        category_description: 'Voici la Desc de la cat3'
    },
    {
        task_id: 6,
        task_title: 'ouloulou',
        task_description: 'zerz erz rsdf s fq zert zer zerzfrez  uyyk wsef',
        duree: 3,
        executor_account_id: 1,
        category_id: 3,
        category_title: 'Category 3',
        category_description: 'Voici la Desc de la cat3'
    },
    {
        task_id: 7,
        task_title: 'fdgh  htfhr t rtdh rthdyt',
        task_description: 'zerz erz rsdf s fq zert zer zerzfrez  uyyk wsef',
        duree: 63,
        executor_account_id: 1,
        category_id: 3,
        category_title: 'Category 3',
        category_description: 'Voici la Desc de la cat3'
    },
    {
        task_id: 8,
        task_title: 'q er gert sg',
        task_description: 'zerz erz rsdf s fq zert zer zerzfrez  uyyk wsef',
        duree: 23,
        executor_account_id: 1,
        category_id: 6,
        category_title: 'dze',
        category_description: 'Voici la Desc de la cat6'
    },
    {
        task_id: 9,
        task_title: 'q er gert sg',
        task_description: 'zerz erz rsdf s fq zert zer zerzfrez  uyyk wsef',
        duree: 23,
        executor_account_id: 2,
        category_id: 6,
        category_title: 'dze',
        category_description: 'Voici la Desc de la cat6'
    },
    // Tâches ajoutées pour respecter 3 tâches par utilisateur et rester sous 160h max par user
    // Pour account_id: 2 (déjà 2 tâches -> en ajouter une)
    {
        task_id: 10,
        task_title: 'Prepare expense report',
        task_description: 'Compiler le rapport mensuel des dépenses.',
        duree: 6,
        executor_account_id: 2,
        category_id: 4,
        category_title: 'Category 4',
        category_description: 'Rapports administratifs'
    },
    // Pour account_id: 3 (déjà 1 tâche -> en ajouter 2)
    {
        task_id: 11,
        task_title: 'Schedule team meeting',
        task_description: 'Planifier et organiser la réunion d’équipe.',
        duree: 55,
        executor_account_id: 3,
        category_id: 1,
        category_title: 'Category 1',
        category_description: 'Tâches administratives basiques'
    },
    {
        task_id: 12,
        task_title: 'Organize office supplies',
        task_description: 'Ranger et inventorier les fournitures de bureau.',
        duree: 31,
        executor_account_id: 3,
        category_id: 5,
        category_title: 'Category 5',
        category_description: 'Gestion de bureau'
    },
    // Pour account_id: 4 (aucune tâche -> en ajouter 3)
    
    {
        task_id: 13,
        task_title: 'Update company calendar',
        task_description: 'Mettre à jour le calendrier de l’entreprise.',
        duree: 52,
        executor_account_id: 4,
        category_id: 1,
        category_title: 'Category 1',
        category_description: 'Tâches administratives basiques'
    },
    {
        task_id: 14,
        task_title: 'File documentation',
        task_description: 'Archiver la documentation administrative.',
        duree: 13,
        executor_account_id: 4,
        category_id: 2,
        category_title: 'Category 2',
        category_description: 'Rapports administratifs'
    },
    {
        task_id: 15,
        task_title: 'Arrange travel itinerary',
        task_description: 'Organiser l’itinéraire de voyage pour les déplacements.',
        duree: 27,
        executor_account_id: 4,
        category_id: 4,
        category_title: 'Category 4',
        category_description: 'Rapports administratifs'
    },
    // Pour account_id: 5 (aucune tâche -> en ajouter 3)
    {
        task_id: 16,
        task_title: 'Process payroll',
        task_description: 'Traiter la paie du personnel.',
        duree: 34,
        executor_account_id: 5,
        category_id: 3,
        category_title: 'Category 3',
        category_description: 'Administration financière'
    },
    {
        task_id: 17,
        task_title: 'Manage invoices',
        task_description: 'Gérer et vérifier les factures fournisseurs.',
        duree: 6,
        executor_account_id: 5,
        category_id: 3,
        category_title: 'Category 3',
        category_description: 'Administration financière'
    },
    {
        task_id: 18,
        task_title: 'Prepare audit documents',
        task_description: 'Rassembler les documents pour l’audit annuel.',
        duree: 44,
        executor_account_id: 5,
        category_id: 2,
        category_title: 'Category 2',
        category_description: 'Rapports administratifs'
    },
    // Pour account_id: 6 (aucune tâche -> en ajouter 3)
    {
        task_id: 19,
        task_title: 'Coordinate staff training',
        task_description: 'Organiser la formation du personnel.',
        duree: 31,
        executor_account_id: 6,
        category_id: 1,
        category_title: 'Category 1',
        category_description: 'Tâches administratives basiques'
    },
    {
        task_id: 20,
        task_title: 'Organize team event',
        task_description: 'Planifier un événement pour l’équipe.',
        duree: 6,
        executor_account_id: 6,
        category_id: 4,
        category_title: 'Category 4',
        category_description: 'Rapports administratifs'
    },
    {
        task_id: 21,
        task_title: 'Update contact directory',
        task_description: 'Mettre à jour le répertoire des contacts.',
        duree: 23,
        executor_account_id: 6,
        category_id: 5,
        category_title: 'Category 5',
        category_description: 'Gestion de bureau'
    },
    // Pour account_id: 7 (aucune tâche -> en ajouter 3)
    {
        task_id: 22,
        task_title: 'Schedule client meeting',
        task_description: 'Organiser une réunion avec un client.',
        duree: 3,
        executor_account_id: 7,
        category_id: 1,
        category_title: 'Category 1',
        category_description: 'Tâches administratives basiques'
    },
    {
        task_id: 23,
        task_title: 'Prepare meeting minutes',
        task_description: 'Rédiger le compte rendu de réunion.',
        duree: 22,
        executor_account_id: 7,
        category_id: 2,
        category_title: 'Category 2',
        category_description: 'Rapports administratifs'
    },
    {
        task_id: 24,
        task_title: 'Compile project summary',
        task_description: 'Compiler le résumé d’un projet en cours.',
        duree: 47,
        executor_account_id: 7,
        category_id: 2,
        category_title: 'Category 2',
        category_description: 'Rapports administratifs'
    },
    // Pour account_id: 8 (aucune tâche -> en ajouter 3)
    {
        task_id: 25,
        task_title: 'Arrange office maintenance',
        task_description: 'Planifier la maintenance des locaux.',
        duree: 45,
        executor_account_id: 8,
        category_id: 5,
        category_title: 'Category 5',
        category_description: 'Gestion de bureau'
    },
    {
        task_id: 26,
        task_title: 'Review compliance documents',
        task_description: 'Vérifier la conformité des documents administratifs.',
        duree: 33,
        executor_account_id: 8,
        category_id: 2,
        category_title: 'Category 2',
        category_description: 'Rapports administratifs'
    },
    {
        task_id: 27,
        task_title: 'Set up conference call',
        task_description: 'Organiser une conférence téléphonique.',
        duree: 24,
        executor_account_id: 8,
        category_id: 1,
        category_title: 'Category 1',
        category_description: 'Tâches administratives basiques'
    },
    // Pour account_id: 9 (aucune tâche -> en ajouter 3)
    {
        task_id: 28,
        task_title: 'Update HR records',
        task_description: 'Mettre à jour les dossiers RH.',
        duree: 2,
        executor_account_id: 9,
        category_id: 5,
        category_title: 'Category 5',
        category_description: 'Gestion de bureau'
    },
    {
        task_id: 29,
        task_title: 'Process leave applications',
        task_description: 'Traiter les demandes de congé.',
        duree: 3,
        executor_account_id: 9,
        category_id: 2,
        category_title: 'Category 2',
        category_description: 'Rapports administratifs'
    },
    {
        task_id: 30,
        task_title: 'Manage vendor contracts',
        task_description: 'Gérer les contrats avec les fournisseurs.',
        duree: 4,
        executor_account_id: 9,
        category_id: 3,
        category_title: 'Category 3',
        category_description: 'Administration financière'
    },
    // Pour account_id: 10 (nouvel utilisateur -> en ajouter 3)
    {
        task_id: 31,
        task_title: 'Organize digital files',
        task_description: 'Ranger et classer les fichiers numériques.',
        duree: 23,
        executor_account_id: 10,
        category_id: 1,
        category_title: 'Category 1',
        category_description: 'Tâches administratives basiques'
    },
    {
        task_id: 32,
        task_title: 'Prepare quarterly budget',
        task_description: 'Préparer le budget trimestriel.',
        duree: 50,
        executor_account_id: 10,
        category_id: 3,
        category_title: 'Category 3',
        category_description: 'Administration financière'
    },
    {
        task_id: 33,
        task_title: 'Review team performance',
        task_description: 'Analyser et synthétiser les performances de l’équipe.',
        duree: 8,
        executor_account_id: 10,
        category_id: 2,
        category_title: 'Category 2',
        category_description: 'Rapports administratifs'
    }
];

export const accountsData: any[] = [
    {
        account_id: 1,
        first_name: 'john',
        last_name: 'doe'
    },
    {
        account_id: 2,
        first_name: 'elisa',
        last_name: 'DoBrazil'
    },
    {
        account_id: 3,
        first_name: 'Malcom',
        last_name: 'X'
    },
    {
        account_id: 4,
        first_name: 'Jean',
        last_name: 'Heude'
    },
    {
        account_id: 5,
        first_name: 'Pi',
        last_name: 'Rogue'
    },
    {
        account_id: 6,
        first_name: 'Alice',
        last_name: 'Dupont'
    },
    {
        account_id: 7,
        first_name: 'Marc',
        last_name: 'Martin'
    },
    {
        account_id: 8,
        first_name: 'Sophie',
        last_name: 'Leroy'
    },
    {
        account_id: 9,
        first_name: 'Lucas',
        last_name: 'Bernard'
    },
    {
        account_id: 10,
        first_name: 'Emma',
        last_name: 'Durand'
    },
];
