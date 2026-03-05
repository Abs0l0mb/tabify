'use strict';

import { PostgresTable } from '@src/classes';

export class AccessRights extends PostgresTable {

    public static MANAGE_TEAMS = 'MANAGE TEAMS';
    public static PROGRESS_REPORT = 'PROGRESS REPORT';

    constructor() {
        
        super('access_rights');
    }
}