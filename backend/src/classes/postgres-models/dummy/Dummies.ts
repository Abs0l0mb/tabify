'use strict';

import { PostgresTable } from '@src/classes';

export class Dummies extends PostgresTable {

    constructor() {
        
        super('dummies');
    }
}