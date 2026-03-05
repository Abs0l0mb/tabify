'use strict';

import { 
    TitledPage,
    VerticalSplit,
    TableRow,
    TeamArchivedTasksTable,
    TaskEventsTable,
    ClientLocation
} from '@src/classes';

export class ArchivedPage extends TitledPage {

    private split: VerticalSplit;

    constructor() {

        super('Archived', 'archived');
        
        this.split = new VerticalSplit(this.content);

        const canManageTeams = ClientLocation.get().api.accountData?.access_right_names?.includes('MANAGE TEAMS');
        const urlParam = ClientLocation.get().router.getParams().teamId;
        const myTeamId = ClientLocation.get().api.accountData.team_id;
        const teamId = canManageTeams && typeof urlParam === 'number' ? urlParam : myTeamId;

        this.init(teamId);
    }

    /*
    **
    **
    */
    private init(teamId: number) : void {

        new TeamArchivedTasksTable(teamId, this.split.topContainer.empty())
        .addClass('light-zone')
        .on('select', (tableRow: TableRow) => {
            new TaskEventsTable(tableRow.rowData.id, this.split.bottomContainer.empty())
            .addClass('light-zone');
        });
    }
}