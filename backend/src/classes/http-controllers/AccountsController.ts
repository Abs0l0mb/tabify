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
    Account,
    AccessRights
} from '@src/classes';

/**
 * Classe `AccountsController`
 *
 * Contrôleur responsable de la gestion des comptes utilisateurs.
 * 
 * Cette classe définit les routes HTTP liées aux comptes :
 * - consultation et mise à jour des comptes (`/account`)
 * - suppression de comptes et de sessions
 * - récupération des droits d’accès disponibles
 * - récupération des comptes pour les rapports de progression
 *
 * Toutes les routes sont protégées par des **middlewares** de la classe `Sessions`
 * (principalement `Sessions.canManageTeams` ou `Sessions.canEditProgressReport`)
 * garantissant que seules les personnes autorisées peuvent y accéder.
 *
 */
export class AccountsController {

    /**
     * Initialise les routes liées aux comptes.
     *
     * @param {BackendHttpServer} server - Serveur HTTP backend sur lequel enregistrer les endpoints.
     *
     * @remarks
     * Les routes définies :
     * - `GET /account` → récupération d’un compte par ID  
     * - `POST /account/update` → mise à jour d’un compte  
     * - `POST /account/delete` → suppression d’un compte  
     * - `GET /account/sessions` → récupération des sessions associées à un compte  
     * - `POST /session/delete` → suppression d’une session  
     * - `GET /access-rights` → liste des droits d’accès  
     * - `GET /progress-report/accounts` → comptes visibles dans les rapports de progression  
     */
    constructor(server: BackendHttpServer) {

        server.get('/account', this.getAccount, Sessions.canManageTeams);
        server.post('/account/update', this.updateAccount, Sessions.canManageTeams);
        server.post('/account/delete', this.deleteAccount, Sessions.canManageTeams);

        server.get('/account/sessions', this.getAccountSessions, Sessions.canManageTeams);
        server.post('/session/delete', this.deleteSession, Sessions.canManageTeams);

        server.get('/access-rights', this.getAccessRights, Sessions.canManageTeams);
    }

    /**
     * Récupère les informations d’un compte à partir de son identifiant.
     *
     * @param {HttpRequest} request - Requête HTTP contenant l’ID du compte (`id`).
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si le compte n’existe pas.
     *
     * @example
     * GET /account?id=42
     */
    private async getAccount(request: HttpRequest, response: HttpResponse): Promise<void> {
        const params = await Parser.parse(request.getParameters(), { id: Parser.integer }, true);
        const account = new Account(params.id);

        await account.load();
        if (!account.data)
            throw new PublicError('account-not-found');

        response.sendSuccessContent(await account.getData());
    }

    /**
     * Met à jour les informations d’un compte existant.
     *
     * @param {HttpRequest} request - Requête contenant les données à mettre à jour.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si le compte n’existe pas ou si l’adresse e-mail est déjà utilisée.
     *
     * @example
     * POST /account/update
     * {
     *   "id": 42,
     *   "email": "user@domain.com",
     *   "firstName": "Alice",
     *   "lastName": "Smith",
     *   "teamId": 3,
     *   "password": "optional",
     *   "accessRights": [1, 2, 3]
     * }
     */
    private async updateAccount(request: HttpRequest, response: HttpResponse): Promise<void> {
        const params = await Parser.parse(request.getParameters(), {
            id: Parser.integer,
            email: Parser.email,
            lastName: Parser.string,
            firstName: Parser.string,
            teamId: Parser.integer,
            password: [Parser.string, 'optional'],
            accessRights: Parser.integerArray,
        }, true);
        
        const account = new Account(params.id);
        await account.load();

        if (!account.data)
            throw new PublicError('account-not-found');

        if (params.email !== account.data.email) {
            const accounts = new Accounts();
            const accountData = await accounts.select({ email: params.email }, true);
            if (accountData)
                throw new PublicError('email@already-exists');
        }

        const data: any = {
            email: params.email,
            last_name: params.lastName,
            first_name: params.firstName,
            team_id: params.teamId
        };

        if (params.password)
            data.password_hash = Sessions.hash(params.password);

        await account.update(data);
        await account.setAccessRights(params.accessRights);

        response.sendSuccessContent();
    }

    /**
     * Supprime un compte utilisateur.
     *
     * @param {HttpRequest} request - Requête HTTP contenant l’ID du compte à supprimer.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si le compte n’existe pas.
     *
     * @example
     * POST /account/delete
     * { "id": 42 }
     */
    private async deleteAccount(request: HttpRequest, response: HttpResponse): Promise<void> {
        const params = await Parser.parse(request.getParameters(), { id: Parser.integer }, true);

        const account = new Account(params.id);
        await account.load();

        if (!account.data)
            throw new PublicError('account-not-found');

        await account.delete();
        response.sendSuccessContent();
    }

    /**
     * Récupère toutes les sessions actives d’un compte.
     *
     * @param {HttpRequest} request - Requête HTTP contenant l’ID du compte.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     *
     * @example
     * GET /account/sessions?id=42
     */
    private async getAccountSessions(request: HttpRequest, response: HttpResponse): Promise<void> {
        const params = await Parser.parse(request.getParameters(), { id: Parser.integer }, true);

        const account = new Account(params.id);
        await account.load();

        response.sendSuccessContent(await account.getSessions());
    }

    /**
     * Supprime une session utilisateur spécifique.
     *
     * @param {HttpRequest} request - Requête HTTP contenant l’ID de la session à supprimer.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si la session n’existe pas.
     *
     * @example
     * POST /session/delete
     * { "id": 101 }
     */
    private async deleteSession(request: HttpRequest, response: HttpResponse): Promise<void> {
        const params = await Parser.parse(request.getParameters(), { id: Parser.integer }, true);

        const session = new Session(params.id);
        await session.load();

        if (!session.data)
            throw new PublicError('session-not-found');

        await session.delete();
        response.sendSuccessContent();
    }

    /**
     * Récupère la liste brute des droits d’accès disponibles dans le système.
     *
     * @param {HttpRequest} request - Requête HTTP.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     *
     * @example
     * GET /access-rights
     */
    private async getAccessRights(request: HttpRequest, response: HttpResponse): Promise<void> {
        const accessRights = new AccessRights();
        response.sendSuccessContent(await accessRights.getRaw());
    }
}