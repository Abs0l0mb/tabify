'use strict';

import { 
    HttpServer,
    HttpCORSHandler,
    Sessions,
    IPCServer,
    IPCClient,
    IPCPorts,
    TasksWebSocketChannel,
    AccountsController,
    AuthenticationController,
    DummiesController,
    BackendIPCServer
} from '@src/classes';

import { ChildProcess, fork } from 'child_process';

/**
 * Classe `BackendHttpServer`
 *
 * Point d’entrée principal du backend.
 * 
 * Cette classe étend `HttpServer` et instancie l’ensemble des composants nécessaires
 * au fonctionnement du serveur : 
 * - configuration CORS,
 * - initialisation des contrôleurs REST,
 * - gestion des communications inter‐processus (IPC),
 * - supervision des sous‐processus,
 * - et gestion des signaux système (arrêt propre).
 * 
 * Chaque instance de `BackendHttpServer` représente un serveur HTTP complet,
 * capable de traiter des requêtes API, d’échanger avec des sous‐processus
 * via IPC, et de gérer un canal WebSocket pour les tâches en temps réel.
 *
 * Le serveur de l'application est instancié dans le fichier backend/src/main.ts
 * 
 * @example
 * ```typescript
 * const server = new BackendHttpServer(8080);
 * ```
 */

export class BackendHttpServer extends HttpServer {

    /**
     * Ensemble des sous‐processus enfants créés par le serveur.
     * Clé : PID du processus enfant.
     */
    private childProcesses: {[pid: number]: ChildProcess} = {};

    /**
     * Serveur IPC (Inter‐Process Communication) permettant d’échanger avec
     * d’autres processus internes (backend ou sous‐processus Node.js).
     */
    private ipcServer: IPCServer;
    
    /**
     * Client IPC utilisé pour communiquer avec le sous‐processus secondaire.
     */
    private subProcessIPCClient: IPCClient;
    
    /**
     * Canal WebSocket permettant la diffusion d’événements temps réel liés aux tâches.
     */
    public readonly tasksWebSocketChannel = new TasksWebSocketChannel();
    
    /**
     * Crée une nouvelle instance du serveur HTTP backend.
     *
     * @param {number} port - Numéro de port sur lequel le serveur écoutera les connexions entrantes.
     *
     * @remarks
     * Le constructeur initialise automatiquement :
     * - la gestion des signaux système (`SIGTERM`) ;
     * - la communication IPC ;
     * - le serveur HTTP et ses contrôleurs.
     */
    constructor(
        private port: number
    ) {

        super();

        this.listenForTermination();

        this.initIPC();
        this.initHttpServer();
    }

    /**
     * Écoute le signal système `SIGTERM` et déclenche un arrêt propre du serveur.
     * 
     * Lorsqu’un signal de terminaison est reçu :
     * - tous les sous‐processus enfants sont arrêtés proprement ;
     * - le processus principal se termine.
     */
    private listenForTermination() : void {

        process.on('SIGTERM', async () => {
            for (let pid in this.childProcesses)
                this.childProcesses[pid].kill('SIGTERM');
            process.exit();
        });
    }

    /**
     * Initialise les communications inter‐processus (IPC).
     *
     * @returns {Promise<void>}
     * 
     * @remarks
     * Crée un serveur IPC (`BackendIPCServer`) pour gérer les communications internes,
     * puis initialise un client IPC pour interagir avec un sous‐processus spécifique.
     */
    private async initIPC() : Promise<void> {

        this.ipcServer = new BackendIPCServer(this);

        this.subProcessIPCClient = await this.initIPCClient(IPCPorts.subProcess(), 'SUB_PROCESS_ENTRY_POINT_JS');
    }

    /**
     * Initialise le serveur HTTP principal :
     * - configure les règles CORS ;
     * - instancie tous les contrôleurs REST du backend ;
     * - démarre l’écoute sur le port spécifié.
     *
     * @returns {Promise<void>}
     *
     * @remarks
     * Il est nécessaire d'instancier les nouveaux contrôleurs créés ici.
     */
    private async initHttpServer() : Promise<void> {

        new HttpCORSHandler(this)
        .allowOrigins([
            'https://tms.jpsigroup.space',
            'https://dev.tms.jpsigroup.space'
        ])
        .allowMethods([
            'OPTIONS',
            'GET',
            'POST'
        ]).allowHeaders([
            Sessions.CSRF_TOKEN_NAME
        ]).allowResponseHeaders([
            Sessions.CSRF_TOKEN_NAME
        ]);

        new AccountsController(this);
        new AuthenticationController(this);
        new DummiesController(this);
        
        this.listen(this.port, '0.0.0.0');
    }

    /**
     * Initialise un client IPC pour communiquer avec un sous‐processus.
     *
     * @param {number} port - Port IPC sur lequel se connecter.
     * @param {string} developmentInstanciationKey - Nom du script à lancer en mode développement.
     * @returns {Promise<IPCClient>} Le client IPC initialisé et connecté.
     *
     * @remarks
     * En mode développement, un sous‐processus est lancé via `fork()`, et le client
     * attend un message `"ready"` avant de se connecter.
     */
    private async initIPCClient(port: number, developmentInstanciationKey: string) : Promise<IPCClient> {
        
        if (process.env.MODE === 'development') {
                
            await new Promise<void>((resolve) => {
                
                const forked = fork(developmentInstanciationKey);

                this.childProcesses[forked.pid!] = forked;

                forked.on('message', function(message: any) {
                    if (message === 'ready')
                        resolve();
                });
            });
        }

        const client = new IPCClient(port);
        client.connect();

        return client;
    }

    /**
     * Retourne le client IPC utilisé pour communiquer avec le sous‐processus.
     *
     * @returns {IPCClient} Instance du client IPC.
     */
    public getSubProcessIPCClient() : IPCClient {

        return this.subProcessIPCClient;
    }
}