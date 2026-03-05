'use strict';

import {
    DataTable,
    Block
} from '@src/classes';

export class TeamArchivedTasksTable extends DataTable {

    constructor(teamId: number, parent: Block) {

        super({
            endpoint: '/fetcher/team/archived-tasks',
            endpointData: {
                teamId: teamId
            },
            selectable: true,
        }, parent);
    }
}