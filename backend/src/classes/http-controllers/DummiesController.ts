'use strict';

import {
    BackendHttpServer,
    HttpRequest,
    HttpResponse,
    Parser,
    PublicError,
    Sessions,
    Dummies,
    Dummy,
} from '@src/classes';

/**
 * Classe `DummiesController`
 *
 * Exemple de contrôleur illustrant la gestion d’une entité simple : **Dummy**.
 *
 * Ce contrôleur montre la structure standard d’un contrôleur backend :
 * - Vérification des autorisations avec les middlewares `Sessions.isConnected`
 * - Validation des paramètres avec `Parser`
 * - Gestion des erreurs avec `PublicError`
 * - Accès à la base via les modèles `Dummy` et `Dummies` 
 *
 * Les routes permettent de lister, créer, mettre à jour et supprimer des enregistrements.
 *
 */
export class DummiesController {

    /**
     * Initialise les routes HTTP liées aux objets *Dummy*.
     *
     * @param {BackendHttpServer} server - Serveur HTTP sur lequel les endpoints sont enregistrés.
     *
     * @remarks
     * Routes disponibles :
     * - `GET /dummy` → Récupère un dummy par son identifiant
     * - `GET /dummies` → Liste tous les dummies
     * - `POST /dummies/add` → Ajoute un nouveau dummy
     * - `POST /dummy/update` → Met à jour un dummy existant
     * - `POST /dummy/delete` → Supprime un dummy
     */
    constructor(private server: BackendHttpServer) {

        server.get('/dummy', this.getDummy, Sessions.isConnected);
        server.get('/dummies', this.getDummies, Sessions.isConnected);
        server.post('/dummies/add', this.addDummy, Sessions.isConnected);
        server.post('/dummy/update', this.updateDummy, Sessions.isConnected);
        server.post('/dummy/delete', this.deleteDummy, Sessions.isConnected);
    }

    /**
     * Récupère un dummy à partir de son identifiant.
     *
     * @param {HttpRequest} request - Requête HTTP contenant le paramètre `id`.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si le dummy n’existe pas.
     *
     * @example
     * GET /dummy?id=5
     */
    private async getDummy(request: HttpRequest, response: HttpResponse): Promise<void> {
        const params = await Parser.parse(request.getParameters(), {
            id: Parser.integer,
        }, true);

        const dummy = new Dummy(params.id);
        await dummy.load();

        if (!dummy.data)
            throw new PublicError('dummy-not-found');

        response.sendSuccessContent(dummy.data);
    }

    /**
     * Récupère la liste complète des dummies.
     *
     * @param {HttpRequest} request - Requête HTTP.
     * @param {HttpResponse} response - Réponse HTTP contenant la liste des dummies.
     *
     * @example
     * GET /dummies
     */
    private async getDummies(request: HttpRequest, response: HttpResponse): Promise<void> {
        const dummies = new Dummies();
        response.sendSuccessContent(await dummies.getRaw());
    }

    /**
     * Ajoute un nouveau dummy dans la base de données.
     *
     * @param {HttpRequest} request - Requête contenant le champ `name` du dummy à créer.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si un dummy portant le même nom existe déjà.
     *
     * @example
     * POST /dummies/add
     * {
     *   "name": "Exemple A"
     * }
     */
    private async addDummy(request: HttpRequest, response: HttpResponse): Promise<void> {
        const params = await Parser.parse(request.getParameters(), {
            name: Parser.string
        }, true);

        const dummies = new Dummies();
        const dummyData = await dummies.select({ name: params.name }, true);

        if (dummyData)
            throw new PublicError('name@already-exists');

        await dummies.insert({ name: params.name });
        response.sendSuccessContent();
    }

    /**
     * Met à jour le nom d’un dummy existant.
     *
     * @param {HttpRequest} request - Requête contenant `id` et `name`.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si le dummy n’existe pas ou si le nom est déjà utilisé.
     *
     * @example
     * POST /dummy/update
     * {
     *   "id": 5,
     *   "name": "Nouveau Nom"
     * }
     */
    private async updateDummy(request: HttpRequest, response: HttpResponse): Promise<void> {
        const params = await Parser.parse(request.getParameters(), {
            id: Parser.integer,
            name: Parser.string
        }, true);
        
        const dummy = new Dummy(params.id);
        await dummy.load();

        if (!dummy.data)
            throw new PublicError('dummy-not-found');

        if (params.name !== dummy.data.name) {
            const dummies = new Dummies();
            const dummyData = await dummies.select({ name: params.name }, true);

            if (dummyData)
                throw new PublicError('name@already-exists');
        }

        await dummy.update({ name: params.name });
        response.sendSuccessContent();
    }

    /**
     * Supprime un dummy à partir de son identifiant.
     *
     * @param {HttpRequest} request - Requête contenant l’identifiant `id` du dummy à supprimer.
     * @param {HttpResponse} response - Réponse HTTP envoyée au client.
     * @throws {PublicError} Si le dummy n’existe pas.
     *
     * @example
     * POST /dummy/delete
     * {
     *   "id": 5
     * }
     */
    private async deleteDummy(request: HttpRequest, response: HttpResponse): Promise<void> {
        const params = await Parser.parse(request.getParameters(), {
            id: Parser.integer
        }, true);

        const dummy = new Dummy(params.id);
        await dummy.load();

        if (!dummy.data)
            throw new PublicError('dummy-not-found');

        await dummy.delete();
        response.sendSuccessContent();
    }
}
