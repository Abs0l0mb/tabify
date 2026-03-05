'use strict';

import { 
    Postgres,
    PostgresTable
} from '@src/classes';

export class Accounts extends PostgresTable {

    constructor() {
        
        super('accounts');
    }
}