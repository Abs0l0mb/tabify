'use strict';

import { 
    TitledPage,
    ClientLocation,
    TeamDocumentsTable
} from '@src/classes';

export class DocumentsPage extends TitledPage {

    constructor() {

        super('Documents', 'documents');
        
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

        new TeamDocumentsTable(teamId, this.content).addClass('light-zone');
    }
}