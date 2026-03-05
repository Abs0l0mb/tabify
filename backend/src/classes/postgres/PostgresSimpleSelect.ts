'use strict';

import {
    Postgres,
    PostgresQueryBinder
} from '@src/classes';

export class PostgresSimpleSelect {
    
    constructor(private conditions: any, private from: string) {
    }

    /*
    **
    **
    */
    public async send() : Promise<any[]> {

        const binder = new PostgresQueryBinder();

        const query = `SELECT * FROM ${this.from} ${this.getWhereClause(binder)}`;

        return await Postgres.getRows(query, binder.getParams());
    }

    /*
    **
    **
    */
    private getWhereClause(binder: PostgresQueryBinder) : string {

        const conditions: any = [];

        for (const column in this.conditions) {
            
            const value = this.conditions[column];
            
            if (value === null || value === 'NULL')
                conditions.push(`${column} IS NULL`);
            else if (value === 'NOT_NULL')
                conditions.push(`${column} IS NOT NULL`);
            else
                conditions.push(`${column} = ${binder.addParam(value)}`);
        }

        return `WHERE ${conditions.join(' AND ')}`;
    }
}