'use strict';

import {
    PostgresTableEntry
} from '@src/classes';

export class Dummy extends PostgresTableEntry {

    constructor(public id: number | null, public data: any | null = null) {
        
        super('dummies', id, data);
    }
}