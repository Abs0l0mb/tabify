'use strict';

import {
    BackendHttpServer,
    HttpRequest,
    HttpResponse,
    Parser,
    PublicError,
    Session,
    Sessions,
    Accounts,
    Account
} from '@src/classes';

/**
 * Classe `AuthenticationController`
 *
 * Contrôleur responsable de l’authentification et de la gestion des sessions utilisateur.
 * 
 * Cette classe définit toutes les routes nécessaires à la connexion, à la déconnexion
 * et à la gestion du compte de l’utilisateur actuellement connecté.
 *
 * Elle interagit étroitement avec la classe `Sessions` pour la vérification des tokens,
 * la gestion des cookies sécurisés et le maintien de la session utilisateur.
 *
 */
export class AuthenticationController {

    /**
     * Initialise les routes d’authentification et de gestion du compte utilisateur.
     *
     * @param {BackendHttpServer} server - Serveur HTTP backend sur lequel les routes sont enregistrées.
     *
     * @remarks
     * Routes disponibles :
     * - `POST /login/prerequisites` : récupère les données nécessaires à la vérification du mot de passe
     * - `POST /login` : connecte un utilisateur
     * - `GET /me` : renvoie les informations du compte connecté
     * - `POST /me/update` : met à jour les informations du compte connecté
     * - `GET /me/sessions` : renvoie les sessions actives du compte
     * - `POST /me/session/delete` : supprime une session spécifique
     * - `POST /logout` : déconnecte l’utilisateur courant
     */
    constructor(server: BackendHttpServer) {
        server.post('/login/prerequisites', this.getLoginPrerequisites, Sessions.isNotConnected);
        server.post('/login', this.login, Sessions.isNotConnected);

        server.get('/me', this.getMe, Sessions.isConnected);
        server.post('/me/update', this.updateMe, Sessions.isConnected);
        server.get('/me/sessions', this.getMySessions, Sessions.isConnected);
        server.post('/me/session/delete', this.deleteMySession, Sessions.isConnected);
        server.post('/logout', this.logout, Sessions.isConnected);
    }

    /**
     * Récupère les données nécessaires à la connexion d’un utilisateur (phase pré-login).
     *
     * @param {HttpRequest} request - Requête contenant l’adresse e-mail de l’utilisateur.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si les identifiants fournis sont inconnus.
     *
     * @example
     * POST /login/prerequisites
     * {
     *   "email": "user@domain.com"
     * }
     */
    private async getLoginPrerequisites(request: HttpRequest, response: HttpResponse): Promise<void> {
        const prerequisites = await (new Sessions()).getLoginPrerequisites(request);

        if (prerequisites)
            response.sendSuccessContent(prerequisites);
        else
            throw new PublicError('password@unknown-credentials');
    }

    /**
     * Authentifie un utilisateur et initialise une nouvelle session.
     *
     * @param {HttpRequest} request - Requête contenant les informations d’authentification.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si les identifiants ne correspondent à aucun compte.
     *
     * @example
     * POST /login
     * {
     *   "email": "user@domain.com",
     *   "passwordHash": "f27a1c..."
     * }
     */
    private async login(request: HttpRequest, response: HttpResponse): Promise<void> {
        if (await (new Sessions()).login(request, response))
            response.sendSuccessContent();
        else
            throw new PublicError('password@unknown-credentials');
    }

    /**
     * Récupère les informations du compte actuellement connecté.
     *
     * @param {HttpRequest} request - Requête HTTP avec session active.
     * @param {HttpResponse} response - Réponse contenant les données du compte.
     * @throws {Error} Si le compte n’existe plus en base de données.
     *
     * @example
     * GET /me
     */
    private async getMe(request: HttpRequest, response: HttpResponse): Promise<void> {
        const account = new Account(request.account?.data.id);
        await account.load();

        if (!account.data)
            throw new Error('account-not-found');

        const data = await account.getData();
        delete data.password_hash;

        response.sendSuccessContent(data);
    }

    /**
     * Met à jour les informations du compte connecté.
     *
     * @param {HttpRequest} request - Requête contenant les nouvelles données utilisateur.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si l’adresse e-mail est déjà utilisée par un autre compte.
     *
     * @example
     * POST /me/update
     * {
     *   "email": "new@domain.com",
     *   "firstName": "Alice",
     *   "lastName": "Smith",
     *   "password": "optional"
     * }
     */
    private async updateMe(request: HttpRequest, response: HttpResponse): Promise<void> {
        const params = await Parser.parse(request.getParameters(), {
            email: Parser.string,
            firstName: Parser.string,
            lastName: Parser.string,
            password: [Parser.password, 'optional']
        }, true);
            
        if (params.email !== request.account?.data.email) {
            const accounts = new Accounts();
            const accountData = await accounts.select({ email: params.email }, true);
            if (accountData)
                throw new PublicError('email@account-already-exists');
        }

        const data: any = {
            email: params.email,
            first_name: params.firstName,
            last_name: params.lastName
        };

        if (params.password)
            data.password_hash = Sessions.hash(params.password);

        await request.account?.update(data);
        response.sendSuccessContent();
    }

    /**
     * Récupère toutes les sessions associées au compte connecté.
     *
     * @param {HttpRequest} request - Requête HTTP.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     *
     * @example
     * GET /me/sessions
     */
    private async getMySessions(request: HttpRequest, response: HttpResponse): Promise<void> {
        response.sendSuccessContent(await request.account?.getSessions());
    }

    /**
     * Supprime une session spécifique appartenant au compte connecté.
     *
     * @param {HttpRequest} request - Requête contenant l’ID de la session à supprimer.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si la session n’existe pas ou n’appartient pas à l’utilisateur courant.
     *
     * @example
     * POST /me/session/delete
     * { "id": 101 }
     */
    private async deleteMySession(request: HttpRequest, response: HttpResponse): Promise<void> {
        const params = await Parser.parse(request.getParameters(), { id: Parser.integer }, true);

        const session = new Session(params.id);
        await session.load();

        if (!session.data)
            throw new PublicError('auth-request-not-found');

        const account = await session.getAccount();

        if (account && session.data.cp_account_id !== account.id)
            throw new PublicError('not-allowed');

        await session.delete();
        response.sendSuccessContent();
    }

    /**
     * Déconnecte l’utilisateur actuellement connecté.
     *
     * @param {HttpRequest} request - Requête HTTP contenant la session active.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     *
     * @example
     * POST /logout
     */
    private async logout(request: HttpRequest, response: HttpResponse): Promise<void> {
        await request.session?.delete();
        response.sendSuccessContent();
    }
}
