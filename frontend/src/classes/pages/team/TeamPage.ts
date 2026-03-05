'use strict';

import { 
    TitledPage,
    ClientLocation,
    TeamAccountsTable
} from '@src/classes';

export class TeamPage extends TitledPage {

    constructor() {

        super('Members', 'members');
        
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

        new TeamAccountsTable(teamId, true, this.content).addClass('light-zone');
    }
}