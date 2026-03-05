"use strict";

import {
    Postgres,
    PostgresQueryBinder
} from '@src/classes';

/**
 * Classe `PostgresTableEntry`
 *
 * Représente une entrée unique (ligne) dans une table PostgreSQL.
 * 
 * Cette classe est généralement utilisée parallèlement avec `PostgresTable`, qui fournit les opérations basiques
 * sur une table complète. `PostgresTableEntry` se concentre sur la gestion d’un enregistrement
 * individuel : chargement, mise à jour et suppression.
 * 
 * Elle peut être étendue pour implémenter une logique spécifique à une table donnée.
 *
 * @example
 * ```typescript
 * const user = new PostgresTableEntry("accounts", 42);
 *
 * // Charger les données de l’utilisateur
 * const found = await user.load();
 * if (found) console.log(user.data.email);
 *
 * // Mettre à jour le nom
 * await user.update({ first_name: "Alice" });
 *
 * // Supprimer la ligne
 * await user.delete();
 * ```
 */
export class PostgresTableEntry {

    /**
     * Crée une instance représentant une ligne spécifique dans une table PostgreSQL.
     *
     * @param {string} table - Nom de la table à laquelle appartient l’entrée.
     * @param {number|null} id - Identifiant unique (`id`) de la ligne concernée.
     * @param {any|null} [data=null] - Données déjà connues ou chargées depuis la base (facultatif).
     */
    constructor(public table: string, public id: number | null, public data: any | null = null) {}

    /**
     * Indique si les données de cette entrée ont été chargées depuis la base de données.
     *
     * @returns {boolean} `true` si les données sont disponibles dans la propriété `data`, sinon `false`.
     *
     * @example
     * ```typescript
     * if (!user.isLoaded()) await user.load();
     * ```
     */
    public isLoaded(): boolean {
        return !!this.data;
    }

    /**
     * Charge la ligne correspondant à l’ID depuis la base de données.
     *
     * Cette méthode effectue une requête SQL `SELECT * FROM <table> WHERE id = <id>`.
     * Si aucune ligne ne correspond, `data` est remis à `null`.
     *
     * @returns {Promise<boolean>} `true` si la ligne a été trouvée et chargée, `false` sinon.
     *
     * @example
     * ```typescript
     * const account = new PostgresTableEntry("accounts", 12);
     * const found = await account.load();
     * if (found) console.log(account.data);
     * ```
     */
    public async load(): Promise<boolean> {
        if (!this.id)
            return false;

        const binder = new PostgresQueryBinder();
        const query = `SELECT * FROM ${this.table} WHERE ID = ${binder.addParam(this.id)}`;
        const result = await Postgres.getRows(query, binder.getParams());

        if (Array.isArray(result) && result.length > 0) {
            this.data = result[0];
            return true;
        } else {
            this.data = null;
            return false;
        }
    }

    /**
     * Met à jour la ligne dans la base de données avec les nouvelles valeurs fournies.
     *
     * Cette méthode exécute une requête `UPDATE` dynamique avec liaison de paramètres sécurisée.
     * Si la ligne est mise à jour avec succès, les données locales (`this.data`) sont également modifiées.
     *
     * @param {object} data - Objet contenant les colonnes à mettre à jour et leurs nouvelles valeurs.
     * @returns {Promise<boolean>} `true` si la mise à jour a été effectuée, `false` sinon.
     *
     * @example
     * ```typescript
     * await account.update({
     *   email: "new.email@example.com",
     *   last_login: new Date()
     * });
     * ```
     */
    public async update(data: any): Promise<boolean> {
        if (!this.id)
            return false;

        const binder = new PostgresQueryBinder();
        const columns: string[] = [];

        for (const key in data) {
            if (data[key] === undefined)
                delete data[key];
        }

        for (const column in data)
            columns.push(`${column} = ${binder.addParam(data[column])}`);

        const query = `UPDATE ${this.table} SET ${columns.join(', ')} WHERE ID = ${binder.addParam(this.id)}`;
        await Postgres.exec(query, binder.getParams());

        for (const key in data)
            this.data[key] = data[key];

        return true;
    }

    /**
     * Méthode protégée appelée avant la suppression d’une entrée.
     *
     * Par défaut, elle ne fait rien, mais peut être surchargée dans une sous-classe
     * pour ajouter une logique de nettoyage (par exemple, suppression de dépendances).
     *
     * @protected
     * @example
     * ```typescript
     * protected async beforeDelete(): Promise<void> {
     *   await this.logDeletion();
     * }
     * ```
     */
    protected async beforeDelete(): Promise<void> {}

    /**
     * Supprime la ligne correspondante de la base de données.
     *
     * Exécute une requête SQL `DELETE FROM <table> WHERE id = <id>`.
     * Appelle la méthode `beforeDelete()` avant la suppression pour permettre
     * une logique additionnelle (par exemple : suppression en cascade, journalisation...).
     *
     * @returns {Promise<void>} Une promesse résolue une fois la suppression terminée.
     *
     * @example
     * ```typescript
     * const account = new Account(42);
     * await account.delete();
     * ```
     */
    public async delete(): Promise<void> {
        if (!this.id)
            return;

        await this.beforeDelete();

        const binder = new PostgresQueryBinder();
        const query = `DELETE FROM ${this.table} WHERE ID = ${binder.addParam(this.id)}`;

        return Postgres.exec(query, binder.getParams());
    }
}
