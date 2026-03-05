'use strict';

import {
    DataTable,
    Block
} from '@src/classes';

export class TeamTasksTable extends DataTable {

    constructor(teamId: number, parent: Block) {

        super({
            endpoint: '/fetcher/team/tasks',
            endpointData: {
                teamId: teamId
            },
            selectable: true
        }, parent);
    }
}