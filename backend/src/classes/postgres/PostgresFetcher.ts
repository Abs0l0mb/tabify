'use strict';

import {
    Postgres,
    Log,
    PostgresQueryBinder
} from '@src/classes';

export type PostgresFetcherStringOperator = '~=' | '=';
export type PostgresFetcherNumberOrDateOperator = '=' | '<' | '>' | '<=' | '>=';
export type PostgresFetcherBooleanOperator = '=';

export interface PostgresFetcherFilter {
    column: string,
    operator: PostgresFetcherStringOperator | PostgresFetcherNumberOrDateOperator | PostgresFetcherBooleanOperator,
    value: string | number | Date | boolean | null
}

export type PostgresFetcherOrderingMethod = 'ASC' | 'DESC';

export interface PostgresFetcherOrdering {
    method?: PostgresFetcherOrderingMethod,
    column: string
}

export interface PostgresFetcherPagination {
    lastId?: number,
    lastOrderingColumnValue?: any,
    quantity?: number
}

export interface PostgresFetcherRequest {
    filters?: PostgresFetcherFilter[],
    ordering?: PostgresFetcherOrdering,
    pagination?: PostgresFetcherPagination
}

export type PostgresFetcherManifestColumnType = 'STRING' | 'NUMBER' | 'DATE' | 'BOOLEAN';

export interface PostgresFetcherManifestColumn {
    name: string,
    description?: string,
    type: PostgresFetcherManifestColumnType,
    hidden?: true
}

export interface PostgresFetcherManifest {
    columns: PostgresFetcherManifestColumn[]
}

export abstract class PostgresFetcher {

    static readonly STRING: PostgresFetcherManifestColumnType = 'STRING';
    static readonly NUMBER: PostgresFetcherManifestColumnType = 'NUMBER';
    static readonly DATE: PostgresFetcherManifestColumnType = 'DATE';
    static readonly BOOLEAN: PostgresFetcherManifestColumnType = 'BOOLEAN';

    static readonly DEFAULT_QUANTITY = 250;

    private manifestColumns: Map<string, PostgresFetcherManifestColumn>;

    constructor(private manifest: PostgresFetcherManifest) {

        this.manifestColumns = this.manifest.columns.reduce((acc, item) => {
            acc.set(item.name, item);
            return acc;
        }, new Map());
    }

    /*
    **
    **
    */
    public getManifest() : PostgresFetcherManifest {
        
        return this.manifest;
    }

    /*
    **
    **
    */
    public async request(request: PostgresFetcherRequest) : Promise<any> {

        try {

            const { query, params } = this.buildQuery(request);
            
            return await Postgres.getRows(query, params);
        }
        catch(error) {

            Log.printError(error);

            return [];
        }
    }
    
    /*
    **
    **
    */
    private getColumnType(column: string) : PostgresFetcherManifestColumnType | null {

        for (const manifestColumn of this.manifest.columns) {

            if (manifestColumn.name === column)
                return manifestColumn.type;
        }

        return null;
    }

    /*
    **
    **
    */
    private checkRequest(request: PostgresFetcherRequest) : PostgresFetcherRequest {

        if (typeof request !== 'object' || request === null)
            throw new Error('Invalid request: request should be an object');
    
        const validatedRequest: PostgresFetcherRequest = {
            filters: []
        };
        
        //=======
        //FILTERS
        //=======

        if (request.filters) {

            if (!Array.isArray(request.filters))
                throw new Error('Invalid filters: filters should be an array');
            
            for (const filter of request.filters) {

                if (typeof filter !== 'object'
                    || filter === null
                    || typeof filter.column !== 'string'
                    || typeof filter.operator !== 'string'
                    || typeof filter.value !== 'string') {
                    throw new Error('Malformed filter');
                }

                const { column, operator, value } = filter;

                let outputValue: any = value;

                const type: PostgresFetcherManifestColumnType | null = this.getColumnType(column);

                try {

                    switch (type) {

                        case PostgresFetcher.STRING: {

                            if (!['~=', '='].includes(operator))
                                throw new Error(`Invalid filter STRING operator "${operator}" on column "${column}"`);

                            outputValue = value.toString();

                            break;
                        }

                        case PostgresFetcher.NUMBER: {

                            if (!['=', '<', '>', '<=', '>='].includes(operator))
                                throw new Error(`Invalid filter NUMBER operator "${operator}" on column "${column}"`);

                            outputValue = new Number(value).valueOf();

                            break;
                        }

                        case PostgresFetcher.DATE: {

                            if (!['=', '<', '>', '<=', '>='].includes(operator))
                                throw new Error(`Invalid filter DATE operator "${operator}" on column "${column}"`);

                            outputValue = new Date(value);

                            break;
                        }

                        case PostgresFetcher.BOOLEAN: {

                            if (operator !== '=')
                                throw new Error(`Invalid filter BOOLEAN operator "${operator}" on column "${column}"`);

                            outputValue = value === 'true' ? true : false;

                            break;
                        }

                        default: {

                            throw new Error('Bad filter column');
                        }
                    }
                }
                catch(error) {

                    Log.printError(error);

                    continue;
                }

                validatedRequest.filters!.push({
                    column: column,
                    operator: operator,
                    value: outputValue
                });
            }
        }
    
        //========
        //ORDERING
        //========

        if (request.ordering) {
            
            try {

                if (typeof request.ordering.column !== 'string')
                    throw new Error('Invalid ordering column: ordering.column should be a string');
                
                if (!this.getColumnType(request.ordering.column))
                    throw new Error('Bad ordering column');

                if (typeof request.ordering.method !== 'string' || !['ASC', 'DESC'].includes(request.ordering.method))
                    throw new Error(`Invalid ordering method: ${request.ordering.method}`);

                validatedRequest.ordering = {
                    column: request.ordering.column,
                    method: request.ordering.method
                };
            }
            catch(error) {

                Log.printError(error);
            }
        }
    
        //==========
        //PAGINATION
        //==========

        if (request.pagination) {

            try {

                if (request.pagination.lastId !== undefined && typeof request.pagination.lastId !== 'number')
                    throw new Error('Invalid pagination lastId: pagination.lastId should be a number');
        
                if (request.pagination.quantity !== undefined && typeof request.pagination.quantity !== 'number')
                    throw new Error('Invalid pagination quantity: pagination.quantity should be a number');
                
                validatedRequest.pagination = {
                    lastId: request.pagination.lastId,
                    lastOrderingColumnValue: request.pagination.lastOrderingColumnValue,
                    quantity: request.pagination.quantity
                };
            }
            catch(error) {

                Log.printError(error);
            }
        }
    
        return validatedRequest;
    }
    
    /*
    **
    **
    */
    public buildQuery(request: PostgresFetcherRequest) : { query: string, params: any[] } {

        request = this.checkRequest(request);

        const binder = new PostgresQueryBinder();

        const where: string[] = [];
        const order: string[] = [];
        let limit: string | null = null;

        //================
        //HANDLING FILTERS
        //================

        if (request.filters && request.filters.length > 0) {

            for (const filter of request.filters) {

                let { column, operator, value } = filter;

                if (value === null) {

                    if (operator === '=')
                        where.push(`"${column}" IS NULL`);
                    else
                        throw new Error('Invalid operator for null value');
                }
                
                else if (typeof value === 'string') {

                    if (operator === '~=')
                        where.push(`lower("${column}") LIKE lower(${binder.addParam(`%${value}%`)})`);
                    else
                        where.push(`"${column}" ${operator} ${binder.addParam(value)}`);
                } 
                
                else if (typeof value === 'number' || value instanceof Date)
                    where.push(`"${column}" ${operator} ${binder.addParam(value)}`);

                else if (typeof value === 'boolean')
                    where.push(`"${column}" ${operator} ${binder.addParam(value)}`);

                else
                    throw new Error(`Unsupported filter value type for column: ${column}`);
            }
        }

        //===================
        //HANDLING PAGINATION
        //===================

        const paginationOperator = request.ordering && request.ordering.method === 'DESC' ? '<' : '>';

        if (request.pagination) {

            if (request.pagination.lastId) {
                
                if (request.ordering && request.ordering.column && request.pagination.lastOrderingColumnValue) {
                    
                    const { column } = request.ordering;

                    where.push(`("${column}", "id") ${paginationOperator} (${binder.addParam(request.pagination.lastOrderingColumnValue)}, ${binder.addParam(request.pagination.lastId)})`);
                }
                else
                    where.push(`"id" ${paginationOperator} ${binder.addParam(request.pagination.lastId)}`);
            }
        }

        //========================
        //HANDLING ORDER BY CLAUSE
        //========================

        if (request.ordering) {

            const { column, method } = request.ordering;

            if (this.manifestColumns.get(column)?.type === PostgresFetcher.STRING) {
                
                if (method === 'ASC') {

                    order.push(`CASE WHEN "${column}" ~ '^[0-9]+$' THEN 0 ELSE 1 END ASC`);
                    order.push(`CASE WHEN "${column}" ~ '^[0-9]+$' THEN "${column}"::bigint ELSE NULL END ASC`);
                    order.push(`"${column}" ASC`);
                }
                else {

                    order.push(`CASE WHEN "${column}" ~ '^[0-9]+$' THEN 1 ELSE 0 END ASC`);
                    order.push(`"${column}" DESC`);
                    order.push(`CASE WHEN "${column}" ~ '^[0-9]+$' THEN "${column}"::bigint ELSE NULL END DESC`);
                }
            }
            else
                order.push(`"${column}" ${method}`);
        }

        order.push(`"id" ${request.ordering && request.ordering.method ? request.ordering.method : 'ASC'}`);

        //=====================
        //HANDLING LIMIT CLAUSE
        //=====================

        if (request.pagination && request.pagination.quantity)
            limit = `${binder.addParam(request.pagination.quantity)}`;
        else
            limit = `${PostgresFetcher.DEFAULT_QUANTITY}`;

        //===========
        //FINAL QUERY
        //===========

        const whereClause = where.length > 0 ? `WHERE ${where.join(' AND ')}` : ``;
        const orderClause = order.length > 0 ? `ORDER BY ${order.join(', ')}` : ``;
        const limitClause = limit !== null ? `LIMIT ${limit}` : ``;

        const query = `SELECT * FROM (${this.getBaseQuery(binder)}) AS subquery ${whereClause} ${orderClause} ${limitClause}`;
        
        return {
            query: query,
            params: binder.getParams()
        };
    }

    /*
    **
    **
    */
    protected abstract getBaseQuery(binder: PostgresQueryBinder) : string;
}