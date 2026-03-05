'use strict';

import { PostgresTableEntry } from '@src/classes';

export class AccessRight extends PostgresTableEntry {

    constructor(public id: number | null, public data: any | null = null) {
        
        super('access_rights', id, data);
    }
}