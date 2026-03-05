'use strict';

import {
    TitledPage,
    TasksKanban,
    ClientLocation
} from '@src/classes';

export class TasksPage extends TitledPage {

    private kanban: TasksKanban;

    constructor() {

        super('Tasks', 'tasks custom-report');
        
        this.content.addClass('light-zone');

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

        this.kanban = new TasksKanban(teamId, this.content);
    }
    
    /*
    **
    **
    */
    public async onLeave() : Promise<void> {
        
        this.kanban.onBeforeDelete();
    }
}