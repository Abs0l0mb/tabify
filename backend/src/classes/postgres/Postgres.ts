'use strict';

import {
    Parser,
    Store
} from '@src/classes';

import pg from 'pg';

export class Postgres {

    static POOL: pg.Pool;
    
    /*
    **
    **
    */
    static async getConfiguration() : Promise<pg.PoolConfig> {

        try {

            return await Parser.parse(Store.get('postgresql'), {
                host: Parser.string,
                user: Parser.string,
                database: Parser.string,
                port: [Parser.integer, 'optional'],
                password: [Parser.string, 'optional'],
                connectionTimeoutMillis: [Parser.integer, 'optional'],
                idleTimeoutMillis: [Parser.integer, 'optional'],
                max: [Parser.integer, 'optional'],
                allowExitOnIdle: [Parser.boolean, 'optional']
            });
        }
        catch(error) {

            throw new Error(`Error when parsing postgresql.json: ${error}`);
        }
    }

    /*
    **
    **
    */
    static async getPool() : Promise<pg.Pool> {

        if (Postgres.POOL)
            return Postgres.POOL;
        else {
            
            Postgres.POOL = new pg.Pool(await Postgres.getConfiguration());

            return Postgres.POOL;
        }
    }

    /*
    **
    **
    */
    static async exec(query: string, params: any = []) : Promise<any> {

        try {
            
            return await (await Postgres.getPool()).query(query, params);
        }
        catch(error) {

            throw new Error(`Error at Postgres.exec: ${error}`);
        }
    }

    /*
    **
    **
    */
    static async getRows(query: string, params: any = []) : Promise<any> {

        const result = await Postgres.exec(query, params);

        if (!Array.isArray(result.rows))
            return [];

        return result.rows;
    }

    /*
    **
    **
    */
    static async getRow(query: string, params: any = []) : Promise<any> {

        let rows = await Postgres.getRows(query, params);

        if (rows.length > 0)
            return rows[0];
        else    
            return {};
    }
}