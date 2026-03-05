'use strict';

import {
    Postgres,
    PostgresQueryBinder,
    PostgresSimpleSelect
} from '@src/classes';

/**
 * Classe `PostgresTable`
 *
 * Représente une table SQL PostgreSQL et fournit un ensemble d’opérations de base
 * pour interagir avec celle-ci : insertion, sélection, suppression, etc.
 *
 * Cette classe sert de fondation à tous les modèles du backend, permettant de
 * manipuler des tables sans écrire directement du SQL brut.
 * 
 * Chaque sous-classe ou instance est liée à une table spécifique via le paramètre `table`.
 *
 * @example
 * ```typescript
 * const accounts = new PostgresTable("accounts");
 *
 * // Insertion
 * const newId = await accounts.insert({
 *   email: "user@example.com",
 *   first_name: "Alice"
 * });
 *
 * // Lecture
 * const users = await accounts.getRaw();
 *
 * // Sélection conditionnelle
 * const alice = await accounts.select({ first_name: "Alice" }, true);
 *
 * // Suppression
 * await accounts.deleteWhere({ id: newId });
 * ```
 */
export class PostgresTable {

    /**
     * Crée une nouvelle instance liée à une table spécifique.
     *
     * @param {string} table - Nom de la table PostgreSQL à manipuler.
     */
    constructor(public table: string) {}

    /**
     * Insère une nouvelle ligne dans la table.
     *
     * Cette méthode construit dynamiquement une requête SQL `INSERT INTO`,
     * en liant les valeurs via un `PostgresQueryBinder` pour éviter les injections SQL.
     *
     * @param {object} data - Objet contenant les colonnes et leurs valeurs à insérer.
     * @returns {Promise<number>} L’identifiant (`id`) de la ligne insérée.
     *
     * @example
     * ```typescript
     * const id = await accounts.insert({
     *   email: "new.user@example.com",
     *   first_name: "John",
     *   last_name: "Doe"
     * });
     * ```
     */
    public async insert(data: any): Promise<number> {
        const binder = new PostgresQueryBinder();

        const columns: string[] = [];
        const values: string[] = [];

        for (let column in data) {
            if (data[column] === undefined)
                delete data[column];
        }

        for (let column in data) {
            columns.push(`${column}`);
            values.push(binder.addParam(data[column]));
        }

        const query = `INSERT INTO ${this.table} (${columns.join(', ')}) VALUES (${values.join(', ')}) RETURNING id`;
        const response = await Postgres.exec(query, binder.getParams());

        return response.rows[0].id;
    }

    /**
     * Récupère toutes les lignes de la table.
     *
     * Effectue une requête SQL équivalente à `SELECT * FROM <table>`.
     *
     * @returns {Promise<any[]>} Liste des enregistrements présents dans la table.
     *
     * @example
     * ```typescript
     * const users = await accounts.getRaw();
     * console.log(users.length); // nombre total de lignes
     * ```
     */
    public async getRaw(): Promise<any> {
        return await Postgres.getRows(`SELECT * FROM ${this.table}`);
    }

    /**
     * Sélectionne des lignes correspondant à des conditions données.
     *
     * Utilise la classe `PostgresSimpleSelect` pour construire et exécuter la requête SQL.
     *
     * @param {object} conditions - Conditions sous forme d’objet `{ colonne: valeur }`.
     * @param {boolean} [singleResult=false] - Si vrai, ne renvoie qu’un seul résultat ou `null`.
     * @returns {Promise<any|any[]>} Le ou les enregistrements correspondant(s) aux conditions.
     *
     * @example
     * ```typescript
     * // Récupère tous les utilisateurs d’une équipe
     * const members = await accounts.select({ team_id: 3 });
     *
     * // Récupère un seul utilisateur
     * const admin = await accounts.select({ email: "admin@site.com" }, true);
     * ```
     */
    public async select(conditions: { [key: string]: any }, singleResult: boolean = false): Promise<any> {
        const query = new PostgresSimpleSelect(conditions, this.table);
        const result = await query.send();

        if (singleResult)
            return result.length > 0 ? result[0] : null;
        else
            return result;
    }

    /**
     * Supprime les lignes correspondant à des conditions spécifiques.
     *
     * Construit une requête `DELETE FROM` sécurisée, avec une gestion spéciale
     * des valeurs `NULL` et `NOT_NULL`.
     *
     * @param {object} data - Conditions sous forme d’objet `{ colonne: valeur }`.
     *   - Si `value` vaut `null` ou `"NULL"`, la condition devient `IS NULL`
     *   - Si `value` vaut `"NOT_NULL"`, la condition devient `IS NOT NULL`
     *   - Sinon, une condition d’égalité est utilisée
     * @param {boolean} [debug=false] - (Optionnel) Si activé, permettrait d’afficher la requête SQL (non utilisé ici).
     * @returns {Promise<void>} Une promesse résolue une fois la suppression effectuée.
     *
     * @example
     * ```typescript
     * // Supprime un utilisateur par ID
     * await accounts.deleteWhere({ id: 12 });
     *
     * // Supprime toutes les entrées sans adresse e-mail
     * await accounts.deleteWhere({ email: null });
     * ```
     */
    public async deleteWhere(data: any, debug: boolean = false): Promise<void> {
        const binder = new PostgresQueryBinder();
        const rawConditions: string[] = [];

        for (const column in data) {
            const value = data[column];
            
            if (value === null || value === 'NULL')
                rawConditions.push(`${column} IS NULL`);
            else if (value === 'NOT_NULL')
                rawConditions.push(`${column} IS NOT NULL`);
            else
                rawConditions.push(`${column} = ${binder.addParam(value)}`);
        }

        const query = `DELETE FROM ${this.table} WHERE ${rawConditions.join(' AND ')}`;
        await Postgres.exec(query, binder.getParams());
    }
}
